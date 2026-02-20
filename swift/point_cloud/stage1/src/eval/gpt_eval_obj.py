# -*- coding: utf-8 -*-
"""
gpt_eval_pred_vs_gt.py

目的：
- 读取 infer_point_ae_qwen3_omni.py 保存的 JSONL 评测结果
- 使用 gpt-5-mini（或更便宜的模型）通过 OpenAI Responses API 判断：
    pred_inject_ae_embedding 与 gt_answer 是否“语义上大致一致”
  （不要求逐字一致，不要求覆盖所有细节；只要核心含义/结论兼容即可）
- 保存：
  1) 每条样本的 judge 结果（JSONL）
  2) summary（总 acc、按 split acc、计数等，JSON）

使用方式（示例）：
- export OPENAI_API_KEY=...
- python gpt_eval_pred_vs_gt.py

注意：
- 为了使用 JSON mode，本脚本会显式要求模型输出 JSON。
- 若输出文件已存在，默认 RESUME=True，会跳过已经评测过的 (split, object_id)。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ====== 需要安装 openai SDK（新版本，支持 Responses API）======
# pip install -U openai
try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


# =========================
# 0) 路径与超参（直接改这里）
# =========================

# 推理脚本输出的 JSONL（输入）
INPUT_JSONL = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/swift/point_cloud/stage1/src/eval/outputs/objaverse/infer_point_ae_qwen3_omni_eval.jsonl"

# 输出（每条样本 judge 结果）
OUTPUT_DIR = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/swift/point_cloud/stage1/src/eval/outputs/objaverse"
OUTPUT_JSONL = os.path.join(OUTPUT_DIR, "gpt_judge_pred_vs_gt.jsonl")
OUTPUT_SUMMARY_JSON = os.path.join(OUTPUT_DIR, "gpt_judge_pred_vs_gt_summary.json")

# 模型：默认 gpt-5-mini（也可改为 gpt-5-nano 更便宜）
JUDGE_MODEL = "gpt-5-mini"

# 是否只评测 status == "ok" 的样本
ONLY_EVAL_STATUS_OK = False

# 是否跳过 pred 为空/None 的样本（若 False，则 pred 为空时直接判错但不调用 API）
SKIP_EMPTY_PRED = False

# 为节省成本：输出 token 上限（判定只需要很短 JSON）
MAX_OUTPUT_TOKENS = 2048

# 温度固定 0，保证判定更稳定
TEMPERATURE = 0.0

# 是否断点续跑（输出文件存在时会读取并跳过已评测条目）
RESUME = True

# 随机抽样（如只想评测一个子集）
MAX_SAMPLES: Optional[int] = None  # 例如 200；None 表示全量

# 指定只评测哪些 split（None 表示不过滤）
SPLIT_FILTER: Optional[List[str]] = None  # 例如 ["train", "val"]


# =========================
# 1) 工具函数
# =========================

def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _iter_jsonl(path: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            yield ln, obj


def _load_done_keys(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    for _, obj in _iter_jsonl(path):
        k = (str(obj.get("split", "")), str(obj.get("object_id", "")))
        done.add(k)
    return done


def _json_extract_first_object(text: str) -> Optional[Dict[str, Any]]:
    """
    兜底解析：从一段文本里抓第一个 {...} 并 json.loads。
    """
    if text is None:
        return None
    t = str(text).strip()
    if t == "":
        return None
    # 优先直接 loads
    try:
        return json.loads(t)
    except Exception:
        pass
    # 正则抓第一个大括号块（非严格，但可用作兜底）
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


@dataclass
class JudgeResult:
    correct: bool
    raw_json_text: str
    error: Optional[str] = None


def build_judge_prompt(question: str, gt: str, pred: str) -> Tuple[str, str]:
    """
    返回 (system, user) 两段 prompt。
    注意：JSON mode 要求上下文里出现 “JSON” 字样。
    """
    system = (
        "You are a strict but fair evaluator designed to output JSON.\n"
        "Task: Given a question, a ground-truth answer, and a model prediction, decide whether the prediction is broadly "
        "consistent with the ground truth.\n\n"
        "Guidelines:\n"
        "- We only require rough semantic consistency (paraphrase is OK).\n"
        "- Do NOT require exact wording, full detail coverage, or perfect completeness.\n"
        "- Mark correct=true if the prediction matches the main object/category/attributes and does not contradict key facts.\n"
        "- Mark correct=false if the prediction describes a different object, contradicts the core meaning, or is clearly wrong.\n"
        "- If the prediction is empty, nonsensical, or an error message, mark correct=false.\n\n"
        "Output strictly valid JSON with exactly this schema:\n"
        '{"correct": true}\n'
        "or\n"
        '{"correct": false}\n'
    )

    user = (
        "Question:\n"
        f"{question}\n\n"
        "Ground truth answer:\n"
        f"{gt}\n\n"
        "Model prediction:\n"
        f"{pred}\n\n"
        "Return JSON only."
    )
    return system, user


def judge_one(
    *,
    client,
    model: str,
    question: str,
    gt: str,
    pred: str,
    temperature: float,
    max_output_tokens: int,
    max_retries: int = 6,
    sleep_base: float = 1.5,
) -> JudgeResult:
    """
    调用 OpenAI Responses API 做一次二分类判定（JSON mode）。
    """
    system, user = build_judge_prompt(question, gt, pred)

    last_err: Optional[str] = None
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                text={"format": {"type": "json_object"}},
                temperature=float(temperature),
                max_output_tokens=int(max_output_tokens),
            )

            # 状态检查：completed 才算成功
            status = getattr(resp, "status", None)
            if status is not None and str(status) != "completed":
                last_err = f"response_status_not_completed: {status}"
                # 仍可尝试解析 output_text（有时 incomplete 也可能含 JSON，但不建议）
                out_text = getattr(resp, "output_text", "") or ""
                parsed = _json_extract_first_object(out_text)
                if isinstance(parsed, dict) and "correct" in parsed:
                    return JudgeResult(correct=bool(parsed.get("correct")), raw_json_text=out_text, error=last_err)
                raise RuntimeError(last_err)

            out_text = getattr(resp, "output_text", "") or ""
            parsed = _json_extract_first_object(out_text)
            if not isinstance(parsed, dict) or "correct" not in parsed:
                raise RuntimeError(f"cannot_parse_json: {out_text[:200]}")

            return JudgeResult(correct=bool(parsed.get("correct")), raw_json_text=out_text, error=None)

        except Exception as e:
            last_err = repr(e)
            # 简单指数退避
            sleep_s = sleep_base * (2 ** attempt) + random.random() * 0.2
            time.sleep(sleep_s)

    return JudgeResult(correct=False, raw_json_text="", error=f"failed_after_retries: {last_err}")


# =========================
# 2) 主流程
# =========================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, default=INPUT_JSONL)
    parser.add_argument("--output_jsonl", type=str, default=OUTPUT_JSONL)
    parser.add_argument("--output_summary", type=str, default=OUTPUT_SUMMARY_JSON)
    parser.add_argument("--model", type=str, default=JUDGE_MODEL)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--only_ok", action="store_true", default=ONLY_EVAL_STATUS_OK)
    parser.add_argument("--resume", action="store_true", default=RESUME)
    parser.add_argument("--no_resume", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--max_output_tokens", type=int, default=MAX_OUTPUT_TOKENS)
    args = parser.parse_args()

    if args.no_resume:
        args.resume = False

    if OpenAI is None:
        raise RuntimeError(
            "openai SDK import failed. Please install/update it:\n"
            "  pip install -U openai\n"
            f"Import error: {repr(_IMPORT_ERROR)}"
        )

    input_jsonl = args.input_jsonl
    output_jsonl = args.output_jsonl
    output_summary = args.output_summary

    if not os.path.exists(input_jsonl):
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")

    _ensure_dir(os.path.dirname(os.path.abspath(output_jsonl)))
    _ensure_dir(os.path.dirname(os.path.abspath(output_summary)))

    # 断点续跑：加载已完成 key
    done_keys = set()
    if args.resume and os.path.exists(output_jsonl):
        done_keys = _load_done_keys(output_jsonl)

    # 读入全部样本（可选抽样）
    items: List[Tuple[int, Dict[str, Any]]] = []
    for ln, obj in _iter_jsonl(input_jsonl):
        if SPLIT_FILTER is not None:
            sp = str(obj.get("split", ""))
            if sp not in SPLIT_FILTER:
                continue
        if args.only_ok:
            if str(obj.get("status", "")) != "ok":
                continue
        items.append((ln, obj))

    if args.max_samples is not None and int(args.max_samples) > 0:
        random.shuffle(items)
        items = items[: int(args.max_samples)]
    elif MAX_SAMPLES is not None and int(MAX_SAMPLES) > 0:
        random.shuffle(items)
        items = items[: int(MAX_SAMPLES)]

    client = OpenAI()

    # 输出文件：w 覆盖 or a 追加（resume 模式一般 append）
    out_mode = "a" if args.resume else "w"
    with open(output_jsonl, out_mode, encoding="utf-8") as wf:
        # 统计
        total = 0
        evaluated = 0
        correct_cnt = 0

        per_split_total: Dict[str, int] = {}
        per_split_correct: Dict[str, int] = {}
        per_split_eval: Dict[str, int] = {}

        for idx, (ln, obj) in enumerate(items):
            split = str(obj.get("split", ""))
            object_id = str(obj.get("object_id", ""))

            key = (split, object_id)
            if args.resume and key in done_keys:
                continue

            total += 1
            per_split_total[split] = per_split_total.get(split, 0) + 1

            question = str(obj.get("question", "") or "")
            gt = str(obj.get("gt_answer", "") or "")
            pred = obj.get("pred_inject_ae_embedding", None)

            if pred is None:
                pred_str = ""
            else:
                pred_str = str(pred)

            # pred 为空：按配置决定是否跳过 or 直接判错
            if pred_str.strip() == "" or gt.strip() == "":
                if SKIP_EMPTY_PRED:
                    rec = {
                        "split": split,
                        "object_id": object_id,
                        "input_line": int(ln),
                        "correct": False,
                        "skipped": True,
                        "skip_reason": "empty_pred_or_gt",
                        "judge_model": args.model,
                        "error": None,
                    }
                    wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    wf.flush()
                    done_keys.add(key)
                    continue
                else:
                    # 不调用 API，直接判错
                    rec = {
                        "split": split,
                        "object_id": object_id,
                        "input_line": int(ln),
                        "correct": False,
                        "skipped": True,
                        "skip_reason": "empty_pred_or_gt",
                        "judge_model": args.model,
                        "error": None,
                    }
                    wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    wf.flush()
                    done_keys.add(key)

                    evaluated += 1
                    per_split_eval[split] = per_split_eval.get(split, 0) + 1
                    # correct 不加
                    continue

            # 调 GPT 判定
            jr = judge_one(
                client=client,
                model=args.model,
                question=question,
                gt=gt,
                pred=pred_str,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
            )

            rec = {
                "split": split,
                "object_id": object_id,
                "input_line": int(ln),
                "correct": bool(jr.correct),
                "skipped": False,
                "judge_model": args.model,
                "judge_raw": jr.raw_json_text,
                "error": jr.error,
            }
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            wf.flush()
            done_keys.add(key)

            evaluated += 1
            per_split_eval[split] = per_split_eval.get(split, 0) + 1

            if jr.correct:
                correct_cnt += 1
                per_split_correct[split] = per_split_correct.get(split, 0) + 1
            else:
                per_split_correct.setdefault(split, per_split_correct.get(split, 0))

            # 简单进度输出
            if (idx + 1) % 10 == 0:
                acc = (correct_cnt / evaluated) if evaluated > 0 else 0.0
                print(f"[{idx+1}/{len(items)}] evaluated={evaluated} acc={acc:.4f}")

    # 汇总
    overall_acc = (correct_cnt / evaluated) if evaluated > 0 else 0.0
    split_stats = {}
    for sp in sorted(set(list(per_split_total.keys()) + list(per_split_eval.keys()))):
        ev = per_split_eval.get(sp, 0)
        cc = per_split_correct.get(sp, 0)
        split_stats[sp] = {
            "total_seen": int(per_split_total.get(sp, 0)),
            "evaluated": int(ev),
            "correct": int(cc),
            "acc": float(cc / ev) if ev > 0 else None,
        }

    summary = {
        "input_jsonl": os.path.abspath(input_jsonl),
        "output_jsonl": os.path.abspath(output_jsonl),
        "output_summary": os.path.abspath(output_summary),
        "judge_model": args.model,
        "only_eval_status_ok": bool(args.only_ok),
        "resume": bool(args.resume),
        "temperature": float(args.temperature),
        "max_output_tokens": int(args.max_output_tokens),
        "counts": {
            "total_processed": int(total),
            "evaluated": int(evaluated),
            "correct": int(correct_cnt),
            "acc": float(overall_acc),
        },
        "by_split": split_stats,
    }

    with open(output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[INFO] GPT judging done.")
    print(f"- Results JSONL: {os.path.abspath(output_jsonl)}")
    print(f"- Summary JSON : {os.path.abspath(output_summary)}")
    print(f"- Overall acc  : {overall_acc:.4f}  ({correct_cnt}/{evaluated})")


if __name__ == "__main__":
    main()
