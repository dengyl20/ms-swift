#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Caption cleaner for point-cloud dataset (JSON list of conversations).

- Reads a dataset JSON file (list[dict]).
- Only rewrites captions where conversation["from"] == "gpt".
- Uses OpenAI Responses API to rewrite captions into compact, content-dense phrases,
  minimizing boilerplate tokens (e.g., "a 3D model of ...").
- Processes captions in batches and sends requests in parallel (ThreadPool).
- Uses Rich progress bar.

Resume / checkpointing:
- Appends completed results to a JSONL checkpoint file:
    {"id": <int>, "clean_caption": <string>, "response_id": <string>, "request_id": <string>, ...}
- On restart, loads checkpoint, applies completed captions to `data`,
  and skips already-done ids. Delete checkpoint to start over.

Safety / cost / "no wasted requests" guarantees:
- Does NOT submit all batches at once (limits in-flight to MAX_WORKERS).
- Disables SDK automatic retries (max_retries=0).
- No retry loop in user code EXCEPT a special-case retry for transient 502 Bad Gateway.
- If client.responses.create raises ANY exception other than 502, the process terminates immediately (os._exit),
  to avoid any further API requests.
- If response JSON parsing fails (JSONDecodeError), the batch is skipped:
  original captions are used as outputs (treated as LLM response) and checkpointed, and a warning is logged.

Requirements:
  pip install --upgrade openai rich

Environment:
  export OPENAI_API_KEY="..."
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import openai
from openai import OpenAI

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# =========================
# User-editable parameters
# =========================

INPUT_JSON_PATH = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/PointLLM/PointLLM/PointLLM_brief_description_660K_filtered.json"
OUTPUT_JSON_PATH = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/PointLLM/PointLLM/PointLLM_brief_description_660K_cleaned.json"

MODEL = "gpt-5.1-chat-latest"

# Parallelism + batching
BATCH_SIZE = 80
MAX_WORKERS = 32

# Safety/cost guardrail: cap output tokens per request
MAX_OUTPUT_TOKENS = 8192

# Prompt caching:
PROMPT_CACHE_KEY_PREFIX = "pc_pointcloud_caption_clean_v1"
PROMPT_CACHE_KEY_SHARDS = 4

# Debug / partial processing:
PROCESS_LIMIT_CAPTIONS = 0  # 0 = process all captions
PREVIEW_PRINT_N = 100       # Print first N before/after pairs

# Stats logging
PRINT_STATS_EVERY_S = 10.0  # set <=0 to disable periodic log lines

# Resume / checkpointing:
# Progress is appended as JSONL lines:
#   {"id": <int>, "clean_caption": <string>, "response_id": <string>, "request_id": <string>, "batch_index": <int>, ...}
# Delete this file to start over from scratch.
CHECKPOINT_JSONL_PATH = OUTPUT_JSON_PATH + ".ckpt.jsonl"
# Optional durability: fsync every N completed batches (0 disables fsync).
CHECKPOINT_FSYNC_EVERY_N_BATCHES = 40

# Special-case retry for transient 502 Bad Gateway:
RETRY_502_INITIAL_SLEEP_S = 5.0
RETRY_502_MAX_SLEEP_S = 60.0

# =========================
# Prompt (static, cacheable)
# =========================

SYSTEM_PROMPT = r"""
You are a dataset caption cleaner for 3D object captions (point-cloud renderings).

Goal
Rewrite each input caption into a SHORT, CONTENT-DENSE, HUMAN-READABLE English noun phrase.
The rewritten caption must preserve the same semantics (objects + attributes) while removing boilerplate,
high-frequency filler, and modality words that do not describe the object(s).

Core constraints
1) Preserve meaning. Do NOT add objects, remove key objects, or invent attributes.
2) Remove modality/boilerplate phrases entirely. Never mention:
   - "3D", "3-D", "three-dimensional"
   - "model", "rendering", "scan", "mesh", "point cloud", "cloud of points"
   - "represented here", "shown", "depicted", "in this scene", "in a 3D setting"
   - generic wrappers like "a 3D model of", "a 3D rendering of", "a 3D object featuring"
3) Minimize uninformative tokens (especially: leading articles, repetitive conjunctions, and filler clauses).
   - Avoid starting with "a/an/the" when possible.
   - Avoid repeated "and" by using commas, semicolons, or slashes.
   - Avoid "of" when possible by rephrasing (noun compounds, re-ordering).
   - Use "with" sparingly (0-1 times). Prefer attribute listing instead.
4) Keep readability (minimum viable). Output should still be understandable to humans/models.
   - Use standard English words.
   - Do NOT output a pure bag-of-words; keep a coherent noun phrase.
5) Formatting rules:
   - Single line.
   - No trailing period.
   - Use semicolons to separate distinct objects/parts.
   - Use slashes for paired colors or alternatives (e.g., "green/white").
   - Use hyphens for compound modifiers (e.g., "rainbow-colored", "toilet-paper-holder").
   - Keep numerals/quantities when present (e.g., "pair", "two", "set of 3").
6) Length target:
   - Typical: 4–14 words.
   - Multi-object captions: keep under ~25 words when possible by compressing lists and removing repeats.

Input/Output contract
You will receive a JSON object:
  {"items":[{"id": <int>, "caption": <string>}, ...]}

Return a JSON object that matches the provided schema.
For every input item, return exactly one output item with the same id and a cleaned caption.

Compression techniques (preferred)
- Delete modality boilerplate: "A 3D model of ..." -> "<object phrase>"
- Replace "X and Y" color pairs with "X/Y".
- Replace repeated attribute words by grouping:
  "blue tray ..., blue ring ..., blue bowl ..." -> "Blue tray ...; blue ring ...; blue bowl ..."
- Convert "accompanied by / alongside / featuring" into compact list separators (;).
- Prefer "pair purple/black swords; white handles" over "A pair of purple and black swords with white handles".

Do NOT do the following
- Do not say "This is ..." or "There is ...".
- Do not mention the point cloud, the prompt, or the dataset.
- Do not add explanations, confidence, or extra commentary.

Examples (Input -> Output)
01. "A green and white rifle." -> "Green/white rifle"
02. "A 3D rendering of a small building with a roof, accompanied by brown and black shelves, a wooden bench, and a brown box." -> "Small building with roof; brown/black shelves; wooden bench; brown box"
03. "A 3D model of a pair of purple and black swords with white handles." -> "Pair purple/black swords; white handles"
04. "A 3D model featuring a small house, island, road with trash, trash pile, boat with trash, ship, boat with a man, and a fish in trash." -> "Small house; island; trash-strewn road; trash pile; boats/ship; man; fish in trash"
05. "A 3D model of a small, grassy hill." -> "Small grassy hill"
06. "3D rendering of a white sofa with wooden legs and frame." -> "White sofa; wooden legs/frame"
07. "3D model of a wooden fence with posts and gate, featuring a sand base." -> "Wooden fence; posts; gate; sand base"
08. "A 3D model of a rainbow-colored mountain." -> "Rainbow-colored mountain"
09. "A 3D object featuring a house with a door, a teal-colored bowl, a room with a blue wall, a curved wall, a pool with sand, a curved wall with a blue door, a wooden boat, and a wooden box." -> "House with door; teal bowl; room with blue wall; curved wall; sandy pool; blue door; wooden boat; wooden box"
10. "A paper clock, brown origami bird, and thin metal clock in a 3D setting." -> "Paper clock; brown origami bird; thin metal clock"
11. "White plastic cylinder resembling a toilet paper holder." -> "White plastic cylinder (toilet-paper-holder)"
"""

# =========================
# Structured output schema
# =========================

OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "integer"},
                    "clean_caption": {"type": "string"},
                },
                "required": ["id", "clean_caption"],
            },
        }
    },
    "required": ["results"],
}

# =========================
# Helpers
# =========================

_console = Console()
_thread_local = threading.local()

_LEADING_BOILERPLATE_RE = re.compile(
    r"^(?:\s*(?:a|an|the)\s+)?"
    r"(?:3\s*d|3d|three[- ]dimensional)\s*"
    r"(?:model|rendering|scan|object)\s*"
    r"(?:of|showing|depicting|featuring)?\s*",
    flags=re.IGNORECASE,
)


def _fatal_exit(msg: str, exc: Exception | None = None) -> None:
    """
    Hard terminate the entire process immediately.
    This is used to guarantee we do not continue making any more API requests.
    """
    try:
        sys.stderr.write("\n[FATAL] " + msg + "\n")
        if exc is not None:
            sys.stderr.write(f"[FATAL] exception: {type(exc).__name__}: {exc}\n")
            # Best-effort: include request_id if present on OpenAI exceptions.
            req_id = getattr(exc, "request_id", None)
            if req_id:
                sys.stderr.write(f"[FATAL] request_id: {req_id}\n")
        sys.stderr.flush()
    finally:
        os._exit(1)


def _get_client() -> OpenAI:
    # Create one OpenAI client per thread to avoid potential thread-safety issues.
    # Also disable SDK automatic retries to avoid duplicate requests.
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = OpenAI(max_retries=0)
        _thread_local.client = client
    return client


def _is_502_bad_gateway(exc: Exception) -> bool:
    """
    Best-effort detection for HTTP 502 Bad Gateway from OpenAI SDK exceptions.
    """
    sc = getattr(exc, "status_code", None)
    if sc == 502:
        return True

    resp = getattr(exc, "response", None)
    if resp is not None:
        sc2 = getattr(resp, "status_code", None)
        if sc2 == 502:
            return True

    # Fallback: match common textual form
    msg = str(exc)
    if "502" in msg and "Bad Gateway" in msg:
        return True

    return False


def _prompt_cache_key(batch_index: int) -> str:
    shard = batch_index % PROMPT_CACHE_KEY_SHARDS
    return f"{PROMPT_CACHE_KEY_PREFIX}_shard{shard:02d}"


def _chunked(items: List[dict], n: int) -> Iterable[List[dict]]:
    for i in range(0, len(items), n):
        yield items[i: i + n]


def _postprocess_caption(s: str) -> str:
    # Very light deterministic cleanup (no fallbacks), to enforce the "no boilerplate" goal.
    s = s.strip()
    s = _LEADING_BOILERPLATE_RE.sub("", s).strip()
    if s.endswith("."):
        s = s[:-1].rstrip()
    return s


def _fmt_int(n: int) -> str:
    # Small, readable formatting for progress display
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.2f}K"
    return str(n)


def _load_checkpoint_jsonl(path: str) -> Dict[int, str]:
    """
    Load JSONL checkpoint: each line is at least {"id": int, "clean_caption": str}.
    Robust to a truncated last line (e.g., interrupted write): stops at first JSON decode error.
    """
    if not path or not os.path.exists(path):
        return {}

    out: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Likely a truncated tail; ignore the rest.
                break
            if not isinstance(obj, dict):
                continue
            if "id" not in obj or "clean_caption" not in obj:
                continue
            try:
                rid = int(obj["id"])
                cap = str(obj["clean_caption"])
            except Exception:
                continue
            out[rid] = cap
    return out


def _append_checkpoint_jsonl(
    fh,
    results: List[dict],
    *,
    response_id: str,
    request_id: str,
    batch_index: int,
    status: str,
    skipped: bool,
    skip_reason: str,
) -> None:
    """
    Append batch results to JSONL checkpoint file handle.
    Adds response_id/request_id so you can find the request/response in OpenAI backend.
    """
    for r in results:
        rec = {
            "id": int(r["id"]),
            "clean_caption": str(r["clean_caption"]),
            "response_id": str(response_id or ""),
            "request_id": str(request_id or ""),
            "batch_index": int(batch_index),
            "status": str(status or ""),
            "skipped": bool(skipped),
            "skip_reason": str(skip_reason or ""),
        }
        fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _extract_usage_dict(resp) -> dict:
    """
    Best-effort extraction of usage. Never raises.
    """
    u = getattr(resp, "usage", None)
    if u is None:
        return {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    input_tokens = _safe_int(getattr(u, "input_tokens", 0), 0)
    output_tokens = _safe_int(getattr(u, "output_tokens", 0), 0)
    total_tokens = _safe_int(getattr(u, "total_tokens", input_tokens + output_tokens), input_tokens + output_tokens)

    cached_tokens = 0
    itd = getattr(u, "input_tokens_details", None)
    if itd is not None:
        if isinstance(itd, dict):
            cached_tokens = _safe_int(itd.get("cached_tokens", 0), 0)
        else:
            cached_tokens = _safe_int(getattr(itd, "cached_tokens", 0), 0)

    return {
        "input_tokens": int(input_tokens),
        "cached_tokens": int(cached_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total_tokens),
    }


@dataclass
class UsageAgg:
    requests: int = 0
    cache_hit_requests: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def add(self, u: dict) -> None:
        self.requests += 1
        self.input_tokens += int(u.get("input_tokens", 0))
        self.cached_tokens += int(u.get("cached_tokens", 0))
        self.output_tokens += int(u.get("output_tokens", 0))
        self.total_tokens += int(u.get("total_tokens", 0))
        if int(u.get("cached_tokens", 0)) > 0:
            self.cache_hit_requests += 1

    def snapshot_fields(self) -> Dict[str, str]:
        in_tok = self.input_tokens
        ct = self.cached_tokens
        req = self.requests

        tok_hit_pct = (100.0 * ct / in_tok) if in_tok > 0 else 0.0
        req_hit_pct = (100.0 * self.cache_hit_requests / req) if req > 0 else 0.0

        return {
            "reqs": str(req),
            "req_hit": f"{req_hit_pct:.1f}%",
            "tok_hit": f"{tok_hit_pct:.1f}%",
            "in_tok": _fmt_int(self.input_tokens),
            "cached_tok": _fmt_int(self.cached_tokens),
            "out_tok": _fmt_int(self.output_tokens),
            "total_tok": _fmt_int(self.total_tokens),
        }

    def log_line(self) -> str:
        f = self.snapshot_fields()
        return (
            f"[stats] req={f['reqs']} | cache_hit(req)={f['req_hit']} | cache_hit(tok)={f['tok_hit']} "
            f"| in={f['in_tok']} (cached {f['cached_tok']}) | out={f['out_tok']} | total={f['total_tok']}"
        )


def _clean_batch(batch_items: List[dict], batch_index: int) -> Tuple[List[dict], dict, dict]:
    """
    batch_items: [{"id": int, "caption": str}, ...]
    returns: (results, usage_dict, meta)
      results: [{"id": int, "clean_caption": str}, ...]  (always one per input item)
      usage_dict: {"input_tokens": int, "cached_tokens": int, "output_tokens": int, "total_tokens": int}
      meta: {"response_id": str, "request_id": str, "status": str, "skipped": bool, "skip_reason": str}
    """
    client = _get_client()

    # Compact JSON input to reduce tokens.
    user_input = json.dumps({"items": batch_items}, ensure_ascii=False, separators=(",", ":"))

    # NOTE: One request per batch max, EXCEPT a special-case retry for transient 502 Bad Gateway.
    sleep_s = RETRY_502_INITIAL_SLEEP_S
    while True:
        try:
            resp = client.responses.create(
                model=MODEL,
                instructions=SYSTEM_PROMPT,
                input=user_input,
                prompt_cache_key=_prompt_cache_key(batch_index),
                max_output_tokens=MAX_OUTPUT_TOKENS,
                store=True,  # explicit, so response_id is retrievable later
                metadata={
                    "job": "pc_caption_clean",
                    "batch_index": str(batch_index),
                    "input_file": os.path.basename(INPUT_JSON_PATH),
                    "output_file": os.path.basename(OUTPUT_JSON_PATH),
                },
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "caption_cleaning",
                        "strict": True,
                        "schema": OUTPUT_JSON_SCHEMA,
                    }
                },
            )
            break
        except Exception as e:
            # Special-case: transient 502 Bad Gateway => sleep then retry.
            if _is_502_bad_gateway(e):
                req_id = getattr(e, "request_id", None)
                _console.log(
                    f"[yellow][retry 502][/yellow] batch_index={batch_index} "
                    f"sleep={sleep_s:.1f}s request_id={req_id or ''}"
                )
                time.sleep(sleep_s)
                sleep_s = min(sleep_s * 2.0, RETRY_502_MAX_SLEEP_S)
                continue

            # Any other failure in create => terminate immediately to avoid further API requests.
            _fatal_exit(f"client.responses.create failed (batch_index={batch_index}). Aborting immediately.", e)

    response_id = str(getattr(resp, "id", "") or "")
    request_id = str(getattr(resp, "_request_id", "") or "")
    status = str(getattr(resp, "status", "") or "")
    usage_dict = _extract_usage_dict(resp)

    skipped = False
    skip_reason = ""

    # Parse model output -> id -> clean_caption map
    out_map: Dict[int, str] = {}

    output_text = getattr(resp, "output_text", None)
    if output_text is None:
        skipped = True
        skip_reason = "missing_output_text"
    else:
        try:
            payload = json.loads(output_text)
            results = payload.get("results", None) if isinstance(payload, dict) else None
            if not isinstance(results, list):
                skipped = True
                skip_reason = "invalid_payload_results"
            else:
                for r in results:
                    if not isinstance(r, dict):
                        continue
                    try:
                        rid = int(r.get("id"))
                        cc = _postprocess_caption(str(r.get("clean_caption", "")))
                    except Exception:
                        continue
                    if cc:
                        out_map[rid] = cc
        except json.JSONDecodeError as e:
            # Skip batch, treat original caption as LLM response, and print/log info.
            skipped = True
            skip_reason = f"JSONDecodeError: {e}"
        except Exception as e:
            # Any non-create parse/format issue: do NOT crash; fallback to originals to avoid re-request on restart.
            skipped = True
            skip_reason = f"parse_error: {type(e).__name__}: {e}"

    # Build final per-item results; always exactly one output per input item.
    final_results: List[dict] = []
    for it in batch_items:
        rid = int(it["id"])
        if rid in out_map:
            cc = out_map[rid]
        else:
            # Treat original caption as if it were the model output; then apply same postprocess.
            cc = _postprocess_caption(str(it.get("caption", "")))
        final_results.append({"id": rid, "clean_caption": cc})

    meta = {
        "response_id": response_id,
        "request_id": request_id,
        "status": status,
        "skipped": bool(skipped),
        "skip_reason": str(skip_reason),
    }
    return final_results, usage_dict, meta


# =========================
# Main
# =========================

def main() -> None:
    if not os.path.exists(INPUT_JSON_PATH):
        raise FileNotFoundError(f"INPUT_JSON_PATH not found: {INPUT_JSON_PATH}")

    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load checkpoint (if any) for resume.
    ckpt_map = _load_checkpoint_jsonl(CHECKPOINT_JSONL_PATH)
    ckpt_ids = set(ckpt_map.keys())
    if ckpt_ids:
        _console.print(
            f"[yellow]Resume enabled:[/yellow] loaded {len(ckpt_ids)} items from checkpoint: {CHECKPOINT_JSONL_PATH}"
        )

    # Collect all GPT captions and remember where they live in the JSON.
    # locations: global_id -> (item_idx, conv_idx)
    locations: Dict[int, Tuple[int, int]] = {}

    # Keep originals only for preview range to avoid huge memory use.
    original: Dict[int, str] = {}

    items_to_clean: List[dict] = []
    gid = 0

    for item_idx, item in enumerate(data):
        conversations = item.get("conversations", [])
        for conv_idx, conv in enumerate(conversations):
            if conv.get("from") != "gpt":
                continue
            cap = conv.get("value", "")
            locations[gid] = (item_idx, conv_idx)
            if gid < PREVIEW_PRINT_N:
                original[gid] = cap
            items_to_clean.append({"id": gid, "caption": cap})

            # If this id is already in checkpoint, apply it immediately to data.
            if gid in ckpt_map:
                data[item_idx]["conversations"][conv_idx]["value"] = _postprocess_caption(ckpt_map[gid])

            gid += 1

    if PROCESS_LIMIT_CAPTIONS and PROCESS_LIMIT_CAPTIONS > 0:
        items_to_clean = items_to_clean[:PROCESS_LIMIT_CAPTIONS]

    scope_total = len(items_to_clean)
    if scope_total == 0:
        _console.print("[yellow]No GPT captions found; nothing to do.[/yellow]")
        return

    # Filter out already-done items within the current scope.
    if ckpt_ids:
        before = len(items_to_clean)
        items_to_clean = [it for it in items_to_clean if int(it["id"]) not in ckpt_ids]
        done_in_scope = before - len(items_to_clean)
    else:
        done_in_scope = 0

    remaining_total = len(items_to_clean)
    if remaining_total == 0:
        _console.print("[green]All captions in current scope are already completed (checkpoint).[/green]")
        # Still write output so the cleaned JSON is materialized.
        os.makedirs(os.path.dirname(OUTPUT_JSON_PATH) or ".", exist_ok=True)
        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        _console.print(f"[green]Done.[/green] Wrote: {OUTPUT_JSON_PATH}")
        return

    batches = list(_chunked(items_to_clean, BATCH_SIZE))

    usage_agg = UsageAgg()
    last_stats_print_t = time.time()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]Cleaning[/bold]"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn(
            " | cache(req) {task.fields[req_hit]} cache(tok) {task.fields[tok_hit]}"
            " | in {task.fields[in_tok]} (c {task.fields[cached_tok]})"
            " | out {task.fields[out_tok]} | tot {task.fields[total_tok]}"
        ),
        console=_console,
    )

    task_id = progress.add_task(
        "clean",
        total=scope_total,
        reqs="0",
        req_hit="0.0%",
        tok_hit="0.0%",
        in_tok="0",
        cached_tok="0",
        out_tok="0",
        total_tok="0",
    )

    # Dispatch batches in parallel with bounded in-flight futures (do NOT submit all at once).
    with progress:
        # Show already-completed progress from checkpoint.
        if done_in_scope > 0:
            progress.update(task_id, advance=done_in_scope)

        os.makedirs(os.path.dirname(CHECKPOINT_JSONL_PATH) or ".", exist_ok=True)
        completed_batches = 0

        # Append-only checkpoint file; each finished batch is appended so resume works after interruption.
        with open(CHECKPOINT_JSONL_PATH, "a", encoding="utf-8") as ckpt_f:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                in_flight: Dict[object, Tuple[int, int]] = {}

                next_to_submit = 0

                def _submit_one(bi: int) -> None:
                    # Submit exactly once; no retries (retry logic, if any, is inside _clean_batch).
                    batch = batches[bi]
                    fut = pool.submit(_clean_batch, batch, bi)
                    in_flight[fut] = (bi, len(batch))

                # Prime the pool
                while next_to_submit < len(batches) and len(in_flight) < MAX_WORKERS:
                    _submit_one(next_to_submit)
                    next_to_submit += 1

                while in_flight:
                    done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                    for fut in done:
                        batch_index, batch_len = in_flight.pop(fut)

                        # If anything unexpected escaped _clean_batch, exit immediately
                        # (avoids continuing to make requests without saving).
                        try:
                            results, usage_dict, meta = fut.result()
                        except Exception as e:
                            _fatal_exit(
                                f"Unexpected worker failure (batch_index={batch_index}). Aborting immediately.",
                                e,
                            )

                        # Log JSON decode skips (and any other skip) with response_id for later lookup.
                        if meta.get("skipped", False):
                            progress.console.log(
                                f"[yellow][skip][/yellow] batch={batch_index} "
                                f"status={meta.get('status','')} "
                                f"response_id={meta.get('response_id','')} request_id={meta.get('request_id','')} "
                                f"reason={meta.get('skip_reason','')}"
                            )

                        # Apply results
                        for r in results:
                            rid = int(r["id"])
                            item_idx, conv_idx = locations[rid]
                            data[item_idx]["conversations"][conv_idx]["value"] = r["clean_caption"]

                        # Persist checkpoint (append-only) so we can resume after interruption.
                        try:
                            _append_checkpoint_jsonl(
                                ckpt_f,
                                results,
                                response_id=meta.get("response_id", ""),
                                request_id=meta.get("request_id", ""),
                                batch_index=batch_index,
                                status=meta.get("status", ""),
                                skipped=bool(meta.get("skipped", False)),
                                skip_reason=str(meta.get("skip_reason", "")),
                            )
                            ckpt_f.flush()
                            completed_batches += 1
                            if CHECKPOINT_FSYNC_EVERY_N_BATCHES and CHECKPOINT_FSYNC_EVERY_N_BATCHES > 0:
                                if (completed_batches % CHECKPOINT_FSYNC_EVERY_N_BATCHES) == 0:
                                    os.fsync(ckpt_f.fileno())
                        except Exception as e:
                            # If we can't write checkpoint, continuing would waste API calls (no durable progress).
                            _fatal_exit("Checkpoint write failed. Aborting immediately to avoid wasting requests.", e)

                        # Update progress + usage/cache stats
                        usage_agg.add(usage_dict)
                        fields = usage_agg.snapshot_fields()
                        progress.update(task_id, advance=batch_len, **fields)

                        # Periodic log line (optional)
                        if PRINT_STATS_EVERY_S and PRINT_STATS_EVERY_S > 0:
                            now = time.time()
                            if (now - last_stats_print_t) >= PRINT_STATS_EVERY_S:
                                progress.console.log(usage_agg.log_line())
                                last_stats_print_t = now

                        # Submit next batch (bounded in-flight)
                        if next_to_submit < len(batches):
                            _submit_one(next_to_submit)
                            next_to_submit += 1

    # Preview a few before/after pairs.
    _console.print("\n[bold]Preview (before -> after):[/bold]")
    for pid in range(min(PREVIEW_PRINT_N, scope_total)):
        item_idx, conv_idx = locations[pid]
        before = original.get(pid, "")
        after = data[item_idx]["conversations"][conv_idx]["value"]
        _console.print(f"[cyan]{pid:>5}[/cyan]  {before}")
        _console.print(f"       -> {after}\n")

    # Final stats summary
    _console.print(f"[bold]Final usage/cache stats:[/bold] {usage_agg.log_line()}")

    # Write output
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH) or ".", exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    _console.print(f"[green]Done.[/green] Wrote: {OUTPUT_JSON_PATH}")
    _console.print(f"[green]Checkpoint:[/green] {CHECKPOINT_JSONL_PATH}")


if __name__ == "__main__":
    main()
