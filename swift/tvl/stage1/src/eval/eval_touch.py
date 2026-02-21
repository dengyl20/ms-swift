#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate existing inference CSV/TSV results with GPT-5 mini judge.

Input rows must contain (either via header or inferred by column count):
  - sample_id
  - question
  - pred
  - gt

This script:
  1) Loads EVAL_PROMPT from util.eval_util to keep judge prompt consistent with your existing pipeline.
  2) Calls OpenAI Responses API with model=gpt-5-mini as a judge.
  3) Appends judge outputs + parsed numeric score into an output CSV/TSV under output_root.
  4) Prints and saves summary (avg score, counts).

Tested assumptions:
  - Your "results.csv" files are actually tab-separated (TSV). Script auto-detects delimiter.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

from openai import OpenAI
EVAL_PROMPT = """[User Question]: {prompt}\n\n
[Assistant Response]: {assistant_response}\n
[Correct Response]: {correct_response}\n\n
We would like to request your feedback on the performance of an AI assistant in response to the user question displayed above. 
The user asks the question on observing an image. The assistant's response is followed by the correct response.
\nPlease evaluate the assistant's response based on how closely it matches the correct response which describes tactile feelings. Please compare only the semantics of the answers. DO NOT consider grammatical errors in scoring the assistant. The assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only one value indicating the score for the assistant. \nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias.\n\n
"""

# -----------------------------
# Utilities
# -----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def detect_delimiter(path: str) -> str:
    """
    Robust delimiter detection for files that may be TSV with .csv extension.
    """
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        sample = f.read(8192)

    # Fast heuristics first
    if "\t" in sample and sample.count("\t") >= sample.count(","):
        return "\t"
    if "," in sample:
        return ","
    if ";" in sample:
        return ";"

    # Fallback
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except Exception:
        return "\t"


def looks_like_header(row: List[str]) -> bool:
    lowered = [c.strip().lower() for c in row]
    return ("sample_id" in lowered) and ("pred" in lowered or "prediction" in lowered) and ("gt" in lowered or "label" in lowered)


def infer_fieldnames_by_len(n: int) -> List[str]:
    """
    For files without header, infer common schemas from your examples.
    """
    # HCT example: 9 columns
    if n == 9:
        return ["sample_id", "dataset", "subset", "source_csv", "tactile", "tactile_background", "question", "pred", "gt"]

    # SSVTP example: 11 columns
    if n == 11:
        return ["sample_id", "dataset", "subset", "source_csv", "tactile", "tactile_background", "question", "pred", "gt", "K_injected", "status"]

    # Unknown: generic
    return [f"col_{i}" for i in range(n)]


def normalize_row_keys(d: Dict[str, str]) -> Dict[str, str]:
    """
    Make sure we can access required fields even if upstream used different key names.
    """
    out = dict(d)

    # Normalize common variants
    # sample_id
    if "sample_id" not in out:
        for k in ["id", "uid", "sample", "sampleid"]:
            if k in out:
                out["sample_id"] = out[k]
                break

    # question
    if "question" not in out:
        for k in ["prompt", "query", "instruction"]:
            if k in out:
                out["question"] = out[k]
                break

    # pred
    if "pred" not in out:
        for k in ["prediction", "generated response", "generated_response", "output", "answer"]:
            if k in out:
                out["pred"] = out[k]
                break

    # gt
    if "gt" not in out:
        for k in ["label", "labels", "ground_truth", "groundtruth", "target"]:
            if k in out:
                out["gt"] = out[k]
                break

    return out


def parse_score_from_judge_text(judge_text: str) -> Optional[float]:
    """
    Try to parse the score the same spirit as your old code: float(evaluation.split()[0]).
    But add robustness for formats like "7/10 ..." or "7.5 - ...".
    """
    if not judge_text:
        return None

    first_token = judge_text.strip().split()[0] if judge_text.strip() else ""
    # Remove common wrappers like "7/10"
    cleaned = re.sub(r"[^0-9.\-]+", "", first_token)
    if cleaned:
        try:
            return float(cleaned)
        except Exception:
            pass

    # Fallback: first number anywhere
    m = re.search(r"(-?\d+(?:\.\d+)?)", judge_text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def load_done_ids(output_path: str, delimiter: str) -> set:
    """
    If output exists, load already-evaluated sample_ids to support resume.
    """
    done = set()
    if not os.path.exists(output_path):
        return done

    with open(output_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if not reader.fieldnames:
            return done
        for row in reader:
            sid = (row.get("sample_id") or "").strip()
            score = (row.get("gpt_judge_score") or "").strip()
            if sid and score:
                done.add(sid)
    return done


# -----------------------------
# GPT judge
# -----------------------------

def get_gpt_evaluator(
    client: OpenAI,
    model_name: str,
    eval_prompt: str,
    max_output_tokens: int = 256,
    reasoning_effort: Optional[str] = "minimal",
    text_verbosity: Optional[str] = "low",
):
    """
    Same callable style as your old pipeline:
      eval_fn(prompt=..., assistant_response=..., correct_response=...) -> str
    """
    system_instructions = "You are a helpful and precise assistant for checking the quality of the answer."

    def evaluate(**kwargs) -> str:
        user_text = eval_prompt.format(**kwargs)

        # Keep request minimal and stable
        payload = dict(
            model=model_name,
            instructions=system_instructions,
            input=user_text,
            max_output_tokens=max_output_tokens,
        )

        # Optional knobs (supported by GPT-5 family)
        # If your org policy/tooling forbids these, you can set them None via CLI.
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}
        if text_verbosity:
            payload["text"] = {"verbosity": text_verbosity}

        resp = client.responses.create(**payload)
        return (resp.output_text or "").strip()

    return evaluate


def call_with_retry(fn, *, max_retries: int = 8, base_sleep: float = 1.0, max_sleep: float = 30.0) -> str:
    """
    Simple exponential backoff retry wrapper for API calls.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            sleep_s = min(max_sleep, base_sleep * (2 ** attempt)) + random.uniform(0, 0.25)
            print(f"[WARN] API call failed (attempt {attempt+1}/{max_retries}): {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(sleep_s)
    raise RuntimeError(f"API call failed after {max_retries} retries: {last_err}") from last_err


# -----------------------------
# Main
# -----------------------------

def process_one_file(
    input_path: str,
    output_path: str,
    model_name: str,
    eval_fn,
    eval_prompt_hash: str,
    only_status: Optional[str],
    max_samples: Optional[int],
) -> Dict:
    delimiter = detect_delimiter(input_path)

    # Resume
    done_ids = load_done_ids(output_path, delimiter=delimiter)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Open input
    with open(input_path, "r", encoding="utf-8", errors="replace", newline="") as fin:
        reader_raw = csv.reader(fin, delimiter=delimiter)
        try:
            first_row = next(reader_raw)
        except StopIteration:
            return {
                "input": input_path,
                "output": output_path,
                "delimiter": delimiter,
                "total_rows": 0,
                "evaluated_rows": 0,
                "skipped_done": 0,
                "skipped_status": 0,
                "scored_rows": 0,
                "avg_score": None,
            }

        if looks_like_header(first_row):
            fieldnames = [c.strip() for c in first_row]
            dict_reader = csv.DictReader(fin, fieldnames=fieldnames, delimiter=delimiter)
        else:
            fieldnames = infer_fieldnames_by_len(len(first_row))
            # Rewind by re-opening and use DictReader with inferred headers
            fin.seek(0)
            dict_reader = csv.DictReader(fin, fieldnames=fieldnames, delimiter=delimiter)

        # Output fieldnames
        extra_cols = [
            "gpt_judge_model",
            "gpt_eval_prompt_sha256",
            "gpt_judge_raw",
            "gpt_judge_score",
            "gpt_judge_error",
            "gpt_judge_timestamp_utc",
        ]

        out_exists = os.path.exists(output_path)
        write_header = True
        if out_exists:
            # If file exists and non-empty, assume header already there.
            if os.path.getsize(output_path) > 0:
                write_header = False

        with open(output_path, "a", encoding="utf-8", newline="") as fout:
            out_fieldnames = fieldnames + [c for c in extra_cols if c not in fieldnames]
            writer = csv.DictWriter(fout, fieldnames=out_fieldnames, delimiter=delimiter)

            if write_header:
                writer.writeheader()

            total_rows = 0
            evaluated_rows = 0
            skipped_done = 0
            skipped_status = 0
            scored_rows = 0
            score_sum = 0.0

            for row in dict_reader:
                total_rows += 1
                if max_samples is not None and evaluated_rows >= max_samples:
                    break

                row = {k: (v if v is not None else "") for k, v in row.items()}
                row = normalize_row_keys(row)

                sid = (row.get("sample_id") or "").strip()
                if sid and sid in done_ids:
                    skipped_done += 1
                    continue

                # Optional: status filtering (useful for SSVTP)
                if only_status is not None:
                    status_val = (row.get("status") or "").strip()
                    # If no status column (e.g., HCT), don't filter it out.
                    if "status" in row and status_val and status_val != only_status:
                        skipped_status += 1
                        continue

                question = (row.get("question") or "").strip()
                pred = (row.get("pred") or "").strip()
                gt = (row.get("gt") or "").strip()

                out_row = dict(row)
                out_row["gpt_judge_model"] = model_name
                out_row["gpt_eval_prompt_sha256"] = eval_prompt_hash
                out_row["gpt_judge_timestamp_utc"] = utc_now_iso()

                if not sid:
                    out_row["gpt_judge_error"] = "missing_sample_id"
                    writer.writerow(out_row)
                    fout.flush()
                    evaluated_rows += 1
                    continue

                if not question or not pred or not gt:
                    out_row["gpt_judge_error"] = f"missing_fields(question={bool(question)}, pred={bool(pred)}, gt={bool(gt)})"
                    writer.writerow(out_row)
                    fout.flush()
                    evaluated_rows += 1
                    continue

                # Call GPT judge
                def _call():
                    return eval_fn(prompt=question, assistant_response=pred, correct_response=gt)

                print(f"[{os.path.basename(input_path)}] Evaluating {sid} ...")
                try:
                    judge_text = call_with_retry(_call)
                    out_row["gpt_judge_raw"] = judge_text
                    score = parse_score_from_judge_text(judge_text)
                    if score is not None:
                        out_row["gpt_judge_score"] = f"{score:.6f}"
                        scored_rows += 1
                        score_sum += float(score)
                    else:
                        out_row["gpt_judge_error"] = "score_parse_failed"
                except Exception as e:
                    out_row["gpt_judge_error"] = f"{type(e).__name__}: {e}"

                writer.writerow(out_row)
                fout.flush()
                evaluated_rows += 1
                if sid:
                    done_ids.add(sid)

            avg_score = (score_sum / scored_rows) if scored_rows > 0 else None

            return {
                "input": input_path,
                "output": output_path,
                "delimiter": delimiter,
                "total_rows": total_rows,
                "evaluated_rows": evaluated_rows,
                "skipped_done": skipped_done,
                "skipped_status": skipped_status,
                "scored_rows": scored_rows,
                "avg_score": avg_score,
            }


def build_output_path(output_root: str, input_path: str) -> str:
    """
    Save results under output_root, keeping subfolders if input is already under output_root.
    Example:
      input:  .../infer_results/hct/results.csv
      output: .../infer_results/hct/results_gpt5mini_eval.csv
    """
    input_path = os.path.abspath(input_path)
    output_root = os.path.abspath(output_root)

    base_dir = os.path.dirname(input_path)
    stem, ext = os.path.splitext(os.path.basename(input_path))
    out_name = f"{stem}_gpt5mini_eval{ext or '.csv'}"

    # If input is under output_root, write alongside it
    try:
        common = os.path.commonpath([output_root, input_path])
    except Exception:
        common = ""

    if common == output_root:
        return os.path.join(base_dir, out_name)

    # Otherwise, put it directly under output_root
    os.makedirs(output_root, exist_ok=True)
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return os.path.join(output_root, f"{safe_stem}_gpt5mini_eval{ext or '.csv'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csvs",
        type=str,
        nargs="+",
        default=[
            "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/tvl_test/outfeatures/infer_results/hct/results.csv",
            "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/tvl_test/outfeatures/infer_results/ssvtp/results.csv",
        ],
        help="One or more inference result CSV/TSV files.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/tvl_test/outfeatures/infer_results",
        help="Root folder to save judged results and summary.",
    )
    parser.add_argument(
        "--gpt_model",
        type=str,
        default="gpt-5-mini",
        help="Judge model id (default: gpt-5-mini).",
    )
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=256,
        help="Max tokens for judge output.",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="minimal",
        choices=["none", "minimal", "low", "medium", "high"],
        help="Reasoning effort for GPT-5 family judge.",
    )
    parser.add_argument(
        "--text_verbosity",
        type=str,
        default="low",
        choices=["low", "medium", "high"],
        help="Verbosity for judge output.",
    )
    parser.add_argument(
        "--only_status",
        type=str,
        default="ok",
        help="If set, only evaluate rows with status == this value (when status column exists).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap: evaluate at most N new samples per input file (useful for quick tests).",
    )

    args = parser.parse_args()

   
    eval_prompt_hash = sha256_text(EVAL_PROMPT)

    # OpenAI client (reads OPENAI_API_KEY from env by default)
    client = OpenAI()

    reasoning_effort = None if args.reasoning_effort == "none" else args.reasoning_effort
    eval_fn = get_gpt_evaluator(
        client=client,
        model_name=args.gpt_model,
        eval_prompt=EVAL_PROMPT,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=reasoning_effort,
        text_verbosity=args.text_verbosity,
    )

    summaries = []
    overall_score_sum = 0.0
    overall_scored = 0

    for input_path in args.input_csvs:
        out_path = build_output_path(args.output_root, input_path)
        summary = process_one_file(
            input_path=input_path,
            output_path=out_path,
            model_name=args.gpt_model,
            eval_fn=eval_fn,
            eval_prompt_hash=eval_prompt_hash,
            only_status=args.only_status,
            max_samples=args.max_samples,
        )
        summaries.append(summary)

        if summary.get("avg_score") is not None and summary.get("scored_rows", 0) > 0:
            # We need weighted average across files
            # We'll approximate by re-using avg*count
            overall_score_sum += float(summary["avg_score"]) * int(summary["scored_rows"])
            overall_scored += int(summary["scored_rows"])

        print("\n=== File Summary ===")
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    overall_avg = (overall_score_sum / overall_scored) if overall_scored > 0 else None
    final_summary = {
        "timestamp_utc": utc_now_iso(),
        "gpt_judge_model": args.gpt_model,
        "eval_prompt_sha256": eval_prompt_hash,
        "files": summaries,
        "overall_scored_rows": overall_scored,
        "overall_avg_score": overall_avg,
    }

    os.makedirs(args.output_root, exist_ok=True)
    summary_path = os.path.join(args.output_root, "gpt5mini_judge_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)

    print("\n=== Overall Summary ===")
    print(json.dumps(final_summary, indent=2, ensure_ascii=False))
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()