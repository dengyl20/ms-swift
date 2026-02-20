# -*- coding: utf-8 -*-
"""
eval_gpt_caption_modelnet40.py

用途：
- 读取你旧脚本生成的 baseline jsonl（old caption prompt 的输出：一句简短描述）
- 调用 OpenAI GPT-5-mini（或其他便宜模型）把 caption 映射到 ModelNet40 的 40 类之一
  * 必须“严格从40类里选一个”，不允许输出“无法判断/unknown/none”
  * 通过 Structured Outputs (JSON Schema + enum) 强制模型输出合法 label
- 将 GPT 的分类结果写入 jsonl
- 计算并输出准确率（同时保存 metrics json）

特点：
- 不用 argparse；全部用全局变量改配置（风格对齐你的旧脚本）
- 可选支持 torchrun 多进程分片（仅依赖 RANK/WORLD_SIZE 环境变量），每个 rank 写 shard，rank0 合并并统计
- 支持断点续跑（RESUME_IF_EXISTS=True 时，会跳过已存在输出行的样本）

依赖：
- pip install openai
- 环境变量：OPENAI_API_KEY

运行示例：
1) 单进程：
   python eval_gpt_caption_modelnet40.py

2) 多进程分片（可选；注意 API rate limit）：
   torchrun --nproc_per_node=4 eval_gpt_caption_modelnet40.py
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# 0) 全局配置（不使用 argparse；直接改这里）
# ============================================================

# ---------- 输入：你旧脚本的 baseline 输出 jsonl ----------
# 旧脚本写的字段名是 baseline_old_prompt_output
BASELINE_CAPTION_JSONL = (
    "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/"
    "ms-swift/swift/point_cloud/stage1/src/eval/modelnet40_infer_outputs/"
    "predictions_baseline_caption_prompt.jsonl"
)

# ---------- (可选) 读取数据集以获得 label 列表（更稳，避免 label 拼写不一致） ----------
USE_DATASET_FOR_LABELS = True
MODELNET40_FEATURE_PT_PATH = (
    "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/"
    "PointLLM/PointLLM/modelnet40_gray_color.pt"
)
REQUIRE_VALID = False  # 仅用于数据集加载时（取 labels）

# ---------- 取样方式（仅两种，和你旧脚本一致） ----------
# "first_n": 从 START_INDEX 起取前 NUM_SAMPLES 条（按 baseline 文件顺序）
# "all": 全量
SAMPLE_MODE = "all"  # "first_n" | "all"
START_INDEX = 0
NUM_SAMPLES = 200  # SAMPLE_MODE="first_n" 时生效

# ---------- 输出 ----------
OUTPUT_DIR = (
    "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/"
    "ms-swift/swift/point_cloud/stage1/src/eval/results/modelnet40_gpt_eval"
)

OUT_GPT_JSONL = os.path.join(OUTPUT_DIR, "gpt_caption_to_modelnet40_preds.jsonl")
OUT_METRICS_JSON = os.path.join(OUTPUT_DIR, "gpt_caption_to_modelnet40_metrics.json")

OVERWRITE_OUTPUT_FILES = True      # True: 覆盖；False: 追加
RESUME_IF_EXISTS = True            # True: 若输出 shard 已存在，则跳过已处理样本
PRINT_EVERY_N = 50                 # 每处理多少条打印一次进度（每个 rank 各自打印）

# ---------- OpenAI / GPT 配置 ----------
OPENAI_MODEL = "gpt-5-mini"        # 也可换 "gpt-5-nano" 更便宜
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"  # 若你用代理/自建网关，可设置此 env
OPENAI_TIMEOUT_SEC = 60.0          # SDK timeout（秒）
OPENAI_SDK_MAX_RETRIES = 0         # SDK 内置重试次数（建议 0，自己做重试更可控）

# GPT 生成参数（分类任务建议尽量确定性）
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_OUTPUT_TOKENS = 32

# Reasoning effort（GPT-5 系列支持 minimal；更快更省）
REASONING_EFFORT = "minimal"       # 可选："minimal" / "low" / "medium" / "high"

# 并发（线程数）；注意别把 API RPM 撑爆
CONCURRENCY = 16

# 失败重试
MAX_CALL_RETRIES = 8
BACKOFF_BASE_SEC = 0.8
BACKOFF_MAX_SEC = 20.0
JITTER_SEC = 0.2

# 若 caption 为空/明显失败标记：是否还要调用 GPT（通常没意义）
CALL_GPT_WHEN_CAPTION_MISSING = False

# 随机种子（仅用于打乱/抽样时；默认不打乱）
SEED = 42

# ============================================================
# 1) 一些常量（最后兜底用的 ModelNet40 40 类）
#    注意：真实 label 以数据集/gt_label 为准；这里仅作为兜底。
# ============================================================

MODELNET40_LABELS_FALLBACK = [
    "airplane",
    "bathtub",
    "bed",
    "bench",
    "bookshelf",
    "bottle",
    "bowl",
    "car",
    "chair",
    "cone",
    "cup",
    "curtain",
    "desk",
    "door",
    "dresser",
    "flower_pot",
    "glass_box",
    "guitar",
    "keyboard",
    "lamp",
    "laptop",
    "mantel",
    "monitor",
    "night_stand",
    "person",
    "piano",
    "plant",
    "radio",
    "range_hood",
    "sink",
    "sofa",
    "stairs",
    "stool",
    "table",
    "tent",
    "toilet",
    "tv_stand",
    "vase",
    "wardrobe",
    "xbox",
]

# ============================================================
# 2) 小工具：RANK/WORLD_SIZE、I/O、JSONL、合并 shard
# ============================================================

def _get_rank_world() -> Tuple[int, int]:
    try:
        rank = int(os.environ.get("RANK", "0"))
    except Exception:
        rank = 0
    try:
        world = int(os.environ.get("WORLD_SIZE", "1"))
    except Exception:
        world = 1
    if world <= 0:
        world = 1
    if rank < 0:
        rank = 0
    if rank >= world:
        rank = rank % world
    return rank, world


def _is_main_process(rank: int, world: int) -> bool:
    return (world <= 1) or (rank == 0)


def _p(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def open_jsonl(path: str, overwrite: bool):
    mode = "w" if overwrite else "a"
    return open(path, mode, encoding="utf-8")


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                # 忽略坏行
                continue


def write_jsonl_line(f, obj: Dict[str, Any]) -> None:
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    f.flush()


def add_rank_suffix(path: str, rank: int) -> str:
    base, ext = os.path.splitext(path)
    return f"{base}.rank{rank}{ext}"


def done_marker_path(out_jsonl_rank: str) -> str:
    return out_jsonl_rank + ".done"


def merge_jsonl_shards(shard_paths: List[str], out_path: str, overwrite: bool) -> None:
    records: List[Dict[str, Any]] = []
    for sp in shard_paths:
        if not os.path.exists(sp):
            continue
        for obj in read_jsonl(sp):
            records.append(obj)

    # 尽量按 run_i / ds_idx 排序恢复原顺序
    def _key(o: Dict[str, Any]) -> Tuple[int, int]:
        a = o.get("run_i", 10**18)
        b = o.get("ds_idx", 10**18)
        try:
            a = int(a)
        except Exception:
            a = 10**18
        try:
            b = int(b)
        except Exception:
            b = 10**18
        return (a, b)

    try:
        records.sort(key=_key)
    except Exception:
        pass

    mode = "w" if overwrite else "a"
    with open(out_path, mode, encoding="utf-8") as fo:
        for obj in records:
            fo.write(json.dumps(obj, ensure_ascii=False) + "\n")
        fo.flush()


def wait_for_all_done(rank_jsonl_paths: List[str]) -> None:
    # rank0 用：等所有 rank 都写完
    done_paths = [done_marker_path(p) for p in rank_jsonl_paths]
    while True:
        if all(os.path.exists(dp) for dp in done_paths):
            return
        time.sleep(2.0)


def load_processed_keys(out_jsonl_rank: str) -> Set[Tuple[int, int]]:
    """
    用于断点续跑：从已有输出文件中读出 (run_i, ds_idx) 键
    """
    keys: Set[Tuple[int, int]] = set()
    if not os.path.exists(out_jsonl_rank):
        return keys
    for obj in read_jsonl(out_jsonl_rank):
        try:
            run_i = int(obj.get("run_i", -1))
            ds_idx = int(obj.get("ds_idx", -1))
            if run_i >= 0 and ds_idx >= 0:
                keys.add((run_i, ds_idx))
        except Exception:
            continue
    return keys


# ============================================================
# 3) 读取 baseline caption records
# ============================================================

@dataclass
class CaptionRecord:
    run_i: int
    ds_idx: int
    gt_label: str
    caption: str
    raw: Dict[str, Any]


def _extract_caption(obj: Dict[str, Any]) -> str:
    # 你旧脚本字段名：baseline_old_prompt_output
    cap = obj.get("baseline_old_prompt_output", "")
    if cap is None:
        cap = ""
    cap = str(cap).strip()
    return cap


def _caption_is_obviously_failed(cap: str) -> bool:
    if cap is None:
        return True
    t = str(cap).strip()
    if t == "":
        return True
    # 旧脚本失败时可能写："[baseline_caption_failed: ...]"
    if t.startswith("[") and "failed" in t.lower():
        return True
    return False


def load_baseline_records(path: str) -> List[CaptionRecord]:
    recs: List[CaptionRecord] = []
    for obj in read_jsonl(path):
        if not isinstance(obj, dict):
            continue
        if "gt_label" not in obj:
            continue
        if "ds_idx" not in obj or "run_i" not in obj:
            continue
        try:
            run_i = int(obj["run_i"])
            ds_idx = int(obj["ds_idx"])
        except Exception:
            continue
        gt = str(obj.get("gt_label", "")).strip()
        cap = _extract_caption(obj)
        recs.append(CaptionRecord(run_i=run_i, ds_idx=ds_idx, gt_label=gt, caption=cap, raw=obj))
    # baseline 文件一般按 run_i 排序；这里稳一点
    recs.sort(key=lambda r: (r.run_i, r.ds_idx))
    return recs


def apply_sample_mode(recs: List[CaptionRecord]) -> List[CaptionRecord]:
    n = len(recs)
    st = max(0, int(START_INDEX))
    if SAMPLE_MODE == "all":
        return recs[st:]
    if SAMPLE_MODE == "first_n":
        end = min(n, st + int(NUM_SAMPLES))
        return recs[st:end]
    raise ValueError(f"Unknown SAMPLE_MODE={SAMPLE_MODE}, must be 'first_n' or 'all'.")


# ============================================================
# 4) 准备 labels（优先从数据集读取；否则从 gt_label 统计；再否则用 fallback）
# ============================================================

def try_load_labels_from_dataset() -> Optional[List[str]]:
    if not USE_DATASET_FOR_LABELS:
        return None
    if not MODELNET40_FEATURE_PT_PATH or (not os.path.exists(MODELNET40_FEATURE_PT_PATH)):
        return None
    try:
        # 与你旧脚本保持一致的 import 路径
        from swift.point_cloud.stage1.src.eval.modelnet40_dataset import ModelNet40PointTokenDataset  # type: ignore

        ds = ModelNet40PointTokenDataset(MODELNET40_FEATURE_PT_PATH, require_valid=REQUIRE_VALID)
        labels = sorted({str(ds[i]["object_labels"]) for i in range(len(ds))})
        if len(labels) > 0:
            return labels
        return None
    except Exception as e:
        _p(f"[WARN] Failed to load labels from dataset: {repr(e)}")
        return None


def build_labels(recs: List[CaptionRecord]) -> List[str]:
    labels = try_load_labels_from_dataset()
    if labels is not None and len(labels) > 0:
        return labels

    # fallback：从 baseline gt_label 里统计（若文件覆盖全 test set，通常也会是 40）
    labels2 = sorted({r.gt_label for r in recs if r.gt_label})
    if len(labels2) > 0:
        return labels2

    return list(MODELNET40_LABELS_FALLBACK)


# ============================================================
# 5) OpenAI 调用：Structured Outputs 强制输出 enum(label)
# ============================================================

def _build_json_schema_for_labels(labels: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": labels,
                "description": "One ModelNet40 category label from the enum.",
            }
        },
        "required": ["label"],
        "additionalProperties": False,
    }


def _build_system_prompt(labels: List[str]) -> str:
    # 不把 labels 逐行塞进 prompt：schema enum 已经提供约束（更省 token）
    # 只强调：必须选一个、不要拒答、不输出 extra
    return (
        "You are a strict classifier.\n"
        "Given a short caption describing an object, select the single most likely ModelNet40 category.\n"
        "You MUST always choose exactly one label from the allowed enum; do not abstain.\n"
        "Do not output anything except the JSON required by the schema.\n"
    )


def _build_user_prompt(caption: str) -> str:
    # caption 可能是英文/中文/噪声文本；原样给
    return (
        "Caption (may be noisy):\n"
        f"{caption.strip()}\n"
    )


def _sleep_with_jitter(base_sec: float) -> None:
    t = float(base_sec)
    if t < 0:
        t = 0.0
    if JITTER_SEC > 0:
        t = t + random.uniform(0.0, float(JITTER_SEC))
    time.sleep(t)


def make_openai_client():
    api_key = os.environ.get(OPENAI_API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Please set env {OPENAI_API_KEY_ENV}=... before running."
        )

    base_url = os.environ.get(OPENAI_BASE_URL_ENV, "").strip() or None

    try:
        import openai  # noqa: F401
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "OpenAI python SDK not installed or import failed. "
            "Please run: pip install -U openai\n"
            f"Import error: {repr(e)}"
        )

    # OpenAI SDK 支持 timeout / max_retries 选项
    kwargs: Dict[str, Any] = {
        "api_key": api_key,
        "timeout": float(OPENAI_TIMEOUT_SEC),
        "max_retries": int(OPENAI_SDK_MAX_RETRIES),
    }
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs)


def call_gpt_label(
    *,
    client,
    system_prompt: str,
    caption: str,
    schema: Dict[str, Any],
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    """
    返回：
      - pred_label: str | None
      - parsed_json: dict | None
      - usage: dict | None
      - error: str | None
    """
    # lazy import errors from SDK（避免没装 openai 时脚本在 import 阶段直接炸）
    import openai  # type: ignore

    user_prompt = _build_user_prompt(caption)

    last_err: Optional[str] = None
    for attempt in range(1, int(MAX_CALL_RETRIES) + 1):
        try:
            resp = client.responses.create(
                model=OPENAI_MODEL,
                input=[
                    {"role": "developer", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                reasoning={"effort": REASONING_EFFORT},
                # temperature=float(TEMPERATURE),
                top_p=float(TOP_P),
                max_output_tokens=int(MAX_OUTPUT_TOKENS),
                # 关键：Structured Outputs，强制 label 从 enum 里选
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "modelnet40_label",
                        "schema": schema,
                        "strict": True,
                    }
                },
                # 不允许工具（避免不必要的 tool call）
                tools=[],
                tool_choice="none",
            )

            out_text = getattr(resp, "output_text", None)
            if out_text is None:
                # 兜底：从 resp.output 拼
                out_text = ""
                try:
                    for item in resp.output:
                        if hasattr(item, "content"):
                            for c in item.content:
                                if hasattr(c, "text"):
                                    out_text += c.text
                except Exception:
                    pass

            out_text = str(out_text).strip()

            parsed = None
            try:
                parsed = json.loads(out_text)
            except Exception:
                # 在 strict schema 下通常不会出现；若出现则当错误重试
                raise RuntimeError(f"JSON parse failed. raw={out_text!r}")

            pred_label = None
            if isinstance(parsed, dict):
                pred_label = parsed.get("label", None)
                if pred_label is not None:
                    pred_label = str(pred_label)

            usage_dict = None
            try:
                u = resp.usage
                usage_dict = {
                    "input_tokens": getattr(u, "input_tokens", None),
                    "output_tokens": getattr(u, "output_tokens", None),
                    "total_tokens": getattr(u, "total_tokens", None),
                }
                # prompt cache details（一般你这个任务 <1024 tokens，不会命中缓存，但留着）
                try:
                    itd = getattr(u, "input_tokens_details", None)
                    if itd is not None:
                        usage_dict["cached_tokens"] = getattr(itd, "cached_tokens", None)
                except Exception:
                    pass
            except Exception:
                usage_dict = None

            return pred_label, parsed, usage_dict, None

        except openai.RateLimitError as e:
            last_err = f"RateLimitError: {repr(e)}"
        except openai.APITimeoutError as e:
            last_err = f"APITimeoutError: {repr(e)}"
        except openai.APIConnectionError as e:
            last_err = f"APIConnectionError: {repr(e)}"
        except openai.APIStatusError as e:
            # 4xx 通常不值得重试；5xx 可以重试
            sc = getattr(e, "status_code", None)
            last_err = f"APIStatusError(status_code={sc}): {repr(e)}"
            if sc is not None and int(sc) < 500:
                break
        except Exception as e:
            last_err = f"Exception: {repr(e)}"

        # backoff
        backoff = min(float(BACKOFF_MAX_SEC), float(BACKOFF_BASE_SEC) * (2.0 ** (attempt - 1)))
        _sleep_with_jitter(backoff)

    return None, None, None, last_err or "unknown_error"


# ============================================================
# 6) 评测与统计
# ============================================================

def compute_metrics_from_predictions_jsonl(
    pred_jsonl_path: str,
    labels: List[str],
) -> Dict[str, Any]:
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    per_tot = [0 for _ in labels]
    per_cor = [0 for _ in labels]

    total = 0
    total_valid_caption = 0
    correct_all = 0
    correct_valid_caption = 0

    n_missing_caption = 0
    n_api_error = 0
    n_pred_none = 0

    for obj in read_jsonl(pred_jsonl_path):
        if not isinstance(obj, dict):
            continue
        if "gt_label" not in obj:
            continue
        total += 1

        gt = str(obj.get("gt_label", "")).strip()
        pred = obj.get("gpt_label", None)
        pred = None if pred is None else str(pred).strip()

        cap_ok = bool(obj.get("caption_valid", False))
        if cap_ok:
            total_valid_caption += 1
        else:
            n_missing_caption += 1

        if gt in label_to_idx:
            per_tot[label_to_idx[gt]] += 1

        err = obj.get("error", None)
        if err:
            n_api_error += 1

        if pred is None or pred == "":
            n_pred_none += 1
            continue

        is_correct = (pred == gt)
        if is_correct:
            correct_all += 1
            if cap_ok:
                correct_valid_caption += 1
            if gt in label_to_idx:
                per_cor[label_to_idx[gt]] += 1

    acc_all = (correct_all / total) if total > 0 else 0.0
    acc_valid = (correct_valid_caption / total_valid_caption) if total_valid_caption > 0 else 0.0

    per_label = {}
    for lab in labels:
        i = label_to_idx[lab]
        tot = per_tot[i]
        cor = per_cor[i]
        per_label[lab] = {
            "total": tot,
            "correct": cor,
            "accuracy": (cor / tot) if tot > 0 else None,
        }

    return {
        "openai_model": OPENAI_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "total_samples": total,
        "total_valid_caption": total_valid_caption,
        "correct_all": correct_all,
        "accuracy_all": acc_all,
        "correct_valid_caption": correct_valid_caption,
        "accuracy_valid_caption_only": acc_valid,
        "missing_caption": n_missing_caption,
        "api_error": n_api_error,
        "pred_label_none": n_pred_none,
        "labels": labels,
        "per_label": per_label,
    }


# ============================================================
# 7) main
# ============================================================

def main() -> None:
    random.seed(SEED)

    rank, world = _get_rank_world()
    is_main = _is_main_process(rank, world)

    if is_main:
        _p(f"[INFO] rank={rank}, world_size={world}")
        _p(f"[INFO] BASELINE_CAPTION_JSONL={BASELINE_CAPTION_JSONL}")
        _p(f"[INFO] OUTPUT_DIR={OUTPUT_DIR}")
        _p(f"[INFO] OPENAI_MODEL={OPENAI_MODEL}, reasoning.effort={REASONING_EFFORT}, CONCURRENCY={CONCURRENCY}")

    if not os.path.exists(BASELINE_CAPTION_JSONL):
        raise FileNotFoundError(f"Baseline jsonl not found: {BASELINE_CAPTION_JSONL}")

    ensure_dir(OUTPUT_DIR)

    # 1) load baseline records
    recs_all = load_baseline_records(BASELINE_CAPTION_JSONL)
    recs = apply_sample_mode(recs_all)

    if is_main:
        _p(f"[INFO] baseline records: total={len(recs_all)}, after sample_mode={len(recs)} (SAMPLE_MODE={SAMPLE_MODE})")

    # 2) labels
    labels = build_labels(recs_all)  # 用全量记录/数据集来拿 label
    if is_main:
        _p(f"[INFO] label_count={len(labels)}")
        if len(labels) != 40:
            _p(f"[WARN] label_count != 40 (got {len(labels)}). Will still evaluate with these labels.")

    schema = _build_json_schema_for_labels(labels)
    system_prompt = _build_system_prompt(labels)

    # 3) distributed sharding by run_i
    #    规则：仅处理 run_i % world == rank
    recs_shard = [r for r in recs if (r.run_i % world) == rank]
    if is_main:
        _p(f"[INFO] shard partition rule: run_i % world == rank")
    _p(f"[INFO][rank{rank}] assigned_samples={len(recs_shard)}")

    # 4) output paths (shard)
    out_rank = add_rank_suffix(OUT_GPT_JSONL, rank) if world > 1 else OUT_GPT_JSONL
    if world > 1:
        _p(f"[INFO][rank{rank}] writing shard: {out_rank}")
    else:
        _p(f"[INFO] writing: {out_rank}")

    # resume
    processed_keys: Set[Tuple[int, int]] = set()
    if RESUME_IF_EXISTS and os.path.exists(out_rank) and (not OVERWRITE_OUTPUT_FILES):
        processed_keys = load_processed_keys(out_rank)
        _p(f"[INFO][rank{rank}] resume enabled. loaded processed_keys={len(processed_keys)}")

    # 5) init OpenAI client
    client = make_openai_client()

    # 6) open output jsonl
    f_out = open_jsonl(out_rank, overwrite=OVERWRITE_OUTPUT_FILES)

    # 7) build tasks
    tasks: List[CaptionRecord] = []
    for r in recs_shard:
        k = (r.run_i, r.ds_idx)
        if RESUME_IF_EXISTS and (not OVERWRITE_OUTPUT_FILES) and (k in processed_keys):
            continue
        tasks.append(r)

    _p(f"[INFO][rank{rank}] pending_tasks={len(tasks)} (after resume-skip)")

    # 8) worker fn
    def _run_one(r: CaptionRecord) -> Dict[str, Any]:
        cap = r.caption
        cap_failed = _caption_is_obviously_failed(cap)
        cap_valid = not cap_failed

        if cap_failed and (not CALL_GPT_WHEN_CAPTION_MISSING):
            return {
                "run_i": r.run_i,
                "ds_idx": r.ds_idx,
                "gt_label": r.gt_label,
                "caption": cap,
                "caption_valid": False,
                "gpt_label": None,
                "correct": False,
                "openai_model": OPENAI_MODEL,
                "error": "caption_missing_or_failed",
            }

        pred_label, parsed_json, usage, err = call_gpt_label(
            client=client,
            system_prompt=system_prompt,
            caption=cap,
            schema=schema,
        )

        correct = (pred_label == r.gt_label) if (pred_label is not None) else False

        out: Dict[str, Any] = {
            "run_i": r.run_i,
            "ds_idx": r.ds_idx,
            "gt_label": r.gt_label,
            "caption": cap,
            "caption_valid": cap_valid,
            "gpt_label": pred_label,
            "correct": correct,
            "openai_model": OPENAI_MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        }
        if usage is not None:
            out["usage"] = usage
        if parsed_json is not None:
            out["gpt_json"] = parsed_json
        if err is not None:
            out["error"] = err
        return out

    # 9) run with threads
    done = 0
    correct_cnt = 0
    valid_caption_cnt = 0
    correct_on_valid_caption = 0
    api_err_cnt = 0
    skipped_missing_caption = 0

    t0 = time.time()
    if CONCURRENCY <= 1:
        for r in tasks:
            obj = _run_one(r)
            write_jsonl_line(f_out, obj)
            done += 1
            if obj.get("caption_valid", False):
                valid_caption_cnt += 1
                if obj.get("correct", False):
                    correct_on_valid_caption += 1
            else:
                skipped_missing_caption += 1

            if obj.get("correct", False):
                correct_cnt += 1
            if obj.get("error", None):
                api_err_cnt += 1

            if (done % int(PRINT_EVERY_N) == 0) or (done == len(tasks)):
                dt = time.time() - t0
                _p(
                    f"[rank{rank}] done={done}/{len(tasks)} "
                    f"acc(all)={(correct_cnt/done):.4f} "
                    f"acc(valid)={(correct_on_valid_caption/valid_caption_cnt if valid_caption_cnt>0 else 0.0):.4f} "
                    f"errors={api_err_cnt} "
                    f"elapsed={dt:.1f}s"
                )
    else:
        with ThreadPoolExecutor(max_workers=int(CONCURRENCY)) as ex:
            futures = {ex.submit(_run_one, r): r for r in tasks}
            for fut in as_completed(futures):
                obj = fut.result()
                write_jsonl_line(f_out, obj)
                done += 1

                if obj.get("caption_valid", False):
                    valid_caption_cnt += 1
                    if obj.get("correct", False):
                        correct_on_valid_caption += 1
                else:
                    skipped_missing_caption += 1

                if obj.get("correct", False):
                    correct_cnt += 1
                if obj.get("error", None):
                    api_err_cnt += 1

                if (done % int(PRINT_EVERY_N) == 0) or (done == len(tasks)):
                    dt = time.time() - t0
                    _p(
                        f"[rank{rank}] done={done}/{len(tasks)} "
                        f"acc(all)={(correct_cnt/done):.4f} "
                        f"acc(valid)={(correct_on_valid_caption/valid_caption_cnt if valid_caption_cnt>0 else 0.0):.4f} "
                        f"errors={api_err_cnt} "
                        f"elapsed={dt:.1f}s"
                    )

    f_out.close()

    # 10) write done marker for distributed merge
    if world > 1:
        with open(done_marker_path(out_rank), "w", encoding="utf-8") as f:
            f.write("done\n")
            f.flush()

    # 11) rank0 merge + metrics
    if world > 1 and is_main:
        shard_paths = [add_rank_suffix(OUT_GPT_JSONL, r) for r in range(world)]
        _p("[INFO][rank0] waiting for all ranks to finish...")
        wait_for_all_done(shard_paths)

        _p("[INFO][rank0] merging shards...")
        merge_jsonl_shards(shard_paths, OUT_GPT_JSONL, overwrite=OVERWRITE_OUTPUT_FILES)
        _p(f"[INFO][rank0] merged to: {OUT_GPT_JSONL}")

        metrics = compute_metrics_from_predictions_jsonl(OUT_GPT_JSONL, labels=labels)
        with open(OUT_METRICS_JSON, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        _p(f"[INFO][rank0] saved metrics: {OUT_METRICS_JSON}")
        _p(
            f"[INFO][rank0] ACC(all)={metrics['accuracy_all']:.6f} "
            f"ACC(valid_caption_only)={metrics['accuracy_valid_caption_only']:.6f} "
            f"total={metrics['total_samples']} valid={metrics['total_valid_caption']} "
            f"missing_caption={metrics['missing_caption']} api_error={metrics['api_error']}"
        )

    if world <= 1:
        metrics = compute_metrics_from_predictions_jsonl(out_rank, labels=labels)
        with open(OUT_METRICS_JSON, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        _p(f"[INFO] saved metrics: {OUT_METRICS_JSON}")
        _p(
            f"[INFO] ACC(all)={metrics['accuracy_all']:.6f} "
            f"ACC(valid_caption_only)={metrics['accuracy_valid_caption_only']:.6f} "
            f"total={metrics['total_samples']} valid={metrics['total_valid_caption']} "
            f"missing_caption={metrics['missing_caption']} api_error={metrics['api_error']}"
        )


if __name__ == "__main__":
    main()
