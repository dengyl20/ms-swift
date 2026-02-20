#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Count how often given ModelNet40 labels appear in TRAIN ground-truth GPT answers.

- TRAIN file format: JSON or JSONL, each sample like:
  {
    "object_id": "...",
    "conversations": [
      {"from": "human", "value": "..."},
      {"from": "gpt", "value": "A blue cartoon"}
    ]
  }

We ONLY count texts where conversations[i]["from"] == "gpt".

Outputs:
- CSV summary with:
  label, train_msg_hits, train_msg_freq, train_occ_hits, train_occ_per_msg, (optional) test_count, test_freq
- JSON detail with same content
"""

from __future__ import annotations

import os
import re
import json
import gzip
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


# =========================
# Global Config (edit here)
# =========================

# Training set file: .json / .jsonl / .json.gz / .jsonl.gz
TRAIN_JSON_PATH = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/PointLLM/PointLLM/PointLLM_brief_description_660K_filtered.json"

# Optional test labels file to restrict/augment stats (leave None if not needed).
# Supported:
#   - .pkl: list[int] or list[str] or dict containing labels
#   - .json: list[int]/list[str]/dict
#   - .txt: one label per line (string)
TEST_LABELS_PATH: Optional[str] = None

# Output paths
OUTPUT_CSV_PATH = "./label_freq_in_train_gt.csv"
OUTPUT_JSON_PATH = "./label_freq_in_train_gt.json"

# If True: only output labels that appear in TEST_LABELS_PATH (if provided)
ONLY_LABELS_IN_TEST = False

# Regex behavior:
# - case-insensitive match
# - allow '_'/' '-'/'space' between multi-token labels (e.g., "flower_pot" matches "flower pot", "flower-pot", "flowerpot")
ALLOW_JOINED_TOKENS = True  # if True, separator can be empty; else require at least one separator


MODELNET40_CLASSES: List[str] = [
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


# =========================
# Utilities
# =========================

def _open_maybe_gzip(path: str, mode: str = "rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def _detect_format(path: str) -> str:
    # return "jsonl" or "json"
    base = path[:-3] if path.endswith(".gz") else path
    if base.endswith(".jsonl"):
        return "jsonl"
    return "json"


def _load_test_labels(path: str) -> List[str]:
    """
    Load test labels as strings.
    Accepts:
      - list[int] -> map via MODELNET40_CLASSES index
      - list[str] -> use directly
      - dict with common keys: labels / y / targets / target / label
    """
    def to_str_labels(obj: Any) -> List[str]:
        if obj is None:
            return []
        if isinstance(obj, dict):
            for k in ["labels", "y", "targets", "target", "label"]:
                if k in obj:
                    return to_str_labels(obj[k])
            # fallback: collect any int/str list inside
            for v in obj.values():
                if isinstance(v, list) and v and isinstance(v[0], (int, str)):
                    return to_str_labels(v)
            return []
        if isinstance(obj, list):
            if not obj:
                return []
            if isinstance(obj[0], int):
                out = []
                for i in obj:
                    if 0 <= i < len(MODELNET40_CLASSES):
                        out.append(MODELNET40_CLASSES[i])
                return out
            if isinstance(obj[0], str):
                return [s.strip() for s in obj if str(s).strip()]
            return []
        return []

    if path.endswith(".pkl") or path.endswith(".pickle"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return to_str_labels(obj)

    if path.endswith(".txt"):
        labels = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    labels.append(s)
        return labels

    # json / json.gz
    with _open_maybe_gzip(path, "rt") as f:
        obj = json.load(f)
    return to_str_labels(obj)


def _iter_train_samples(path: str) -> Iterable[Dict[str, Any]]:
    fmt = _detect_format(path)
    if fmt == "jsonl":
        with _open_maybe_gzip(path, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    else:
        # json: typically a list of dict
        with _open_maybe_gzip(path, "rt") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            for it in obj:
                if isinstance(it, dict):
                    yield it
        elif isinstance(obj, dict):
            # possible wrapper dict, try common keys
            for k in ["data", "samples", "items", "train", "dataset"]:
                v = obj.get(k)
                if isinstance(v, list):
                    for it in v:
                        if isinstance(it, dict):
                            yield it
                    return
            # fallback: treat dict values as samples
            for v in obj.values():
                if isinstance(v, dict):
                    yield v


def _extract_gpt_texts(sample: Dict[str, Any]) -> List[str]:
    convs = sample.get("conversations", None)
    if not isinstance(convs, list):
        return []
    outs: List[str] = []
    for msg in convs:
        if not isinstance(msg, dict):
            continue
        if str(msg.get("from", "")).strip().lower() == "gpt":
            val = msg.get("value", "")
            if val is None:
                continue
            s = str(val).strip()
            if s:
                outs.append(s)
    return outs


def _build_label_pattern(label: str) -> str:
    """
    Build a robust pattern for one label.
    - boundaries based on alnum only (avoid partial matches inside words/numbers)
    - handle multi-token labels with '_' by allowing [_\\s-]* between tokens
    """
    tokens = label.split("_")
    tokens_esc = [re.escape(t) for t in tokens if t]
    if not tokens_esc:
        tokens_esc = [re.escape(label)]

    if len(tokens_esc) == 1:
        core = tokens_esc[0]
    else:
        sep = r"[_\s-]*" if ALLOW_JOINED_TOKENS else r"[_\s-]+"
        core = sep.join(tokens_esc)

    # Use "alnum boundary" (underscore is NOT alnum, so "tv_stand" works)
    # Prevent matching inside longer alnum strings.
    return rf"(?<![A-Za-z0-9]){core}(?![A-Za-z0-9])"


@dataclass
class LabelStats:
    label: str
    train_msg_hits: int = 0     # number of GPT messages that mention label (>=1 match)
    train_occ_hits: int = 0     # total number of matches across all GPT messages

    test_count: Optional[int] = None  # optional
    test_freq: Optional[float] = None # optional


def _save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    import csv
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    labels_interest: List[str] = list(MODELNET40_CLASSES)

    test_labels: Optional[List[str]] = None
    test_label_counts: Dict[str, int] = {}
    if TEST_LABELS_PATH:
        test_labels = _load_test_labels(TEST_LABELS_PATH)
        for lb in test_labels:
            lb_norm = lb.strip()
            if lb_norm:
                test_label_counts[lb_norm] = test_label_counts.get(lb_norm, 0) + 1

        if ONLY_LABELS_IN_TEST:
            # keep order according to MODELNET40_CLASSES if possible, else append extras
            keep = set(test_label_counts.keys())
            labels_interest = [lb for lb in MODELNET40_CLASSES if lb in keep]
            extras = [lb for lb in test_label_counts.keys() if lb not in set(MODELNET40_CLASSES)]
            labels_interest += sorted(extras)

    # Build a single combined regex with named groups, so we scan each GPT message once.
    group_to_label: Dict[str, str] = {}
    parts: List[str] = []
    for i, lb in enumerate(labels_interest):
        gname = f"L{i}"
        group_to_label[gname] = lb
        parts.append(rf"(?P<{gname}>{_build_label_pattern(lb)})")
    combined_re = re.compile("|".join(parts), flags=re.IGNORECASE)

    stats: Dict[str, LabelStats] = {lb: LabelStats(label=lb) for lb in labels_interest}

    total_gpt_msgs = 0

    for sample in _iter_train_samples(TRAIN_JSON_PATH):
        gpt_texts = _extract_gpt_texts(sample)
        for txt in gpt_texts:
            total_gpt_msgs += 1
            found_in_msg: Dict[str, int] = {}  # label -> occ in this msg
            for m in combined_re.finditer(txt):
                g = m.lastgroup
                if not g:
                    continue
                lb = group_to_label[g]
                found_in_msg[lb] = found_in_msg.get(lb, 0) + 1

            for lb, occ in found_in_msg.items():
                stats[lb].train_msg_hits += 1
                stats[lb].train_occ_hits += occ

    # Fill test stats if available
    if test_labels is not None:
        test_total = sum(test_label_counts.values())
        for lb in labels_interest:
            c = test_label_counts.get(lb, 0)
            stats[lb].test_count = c
            stats[lb].test_freq = (c / test_total) if test_total > 0 else 0.0

    # Prepare output rows
    rows: List[Dict[str, Any]] = []
    for lb in labels_interest:
        st = stats[lb]
        train_msg_freq = (st.train_msg_hits / total_gpt_msgs) if total_gpt_msgs > 0 else 0.0
        train_occ_per_msg = (st.train_occ_hits / total_gpt_msgs) if total_gpt_msgs > 0 else 0.0

        row = {
            "label": lb,
            "train_total_gpt_msgs": total_gpt_msgs,
            "train_msg_hits": st.train_msg_hits,
            "train_msg_freq": train_msg_freq,
            "train_occ_hits": st.train_occ_hits,
            "train_occ_per_msg": train_occ_per_msg,
        }
        if st.test_count is not None:
            row["test_count"] = st.test_count
            row["test_freq"] = st.test_freq
        rows.append(row)

    # Sort by train_msg_freq desc (more intuitive for "出现频率")
    rows.sort(key=lambda r: (r["train_msg_freq"], r["train_msg_hits"]), reverse=True)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH) or ".", exist_ok=True)
    _save_csv(rows, OUTPUT_CSV_PATH)

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH) or ".", exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # Print a short summary
    print(f"[OK] TRAIN: {TRAIN_JSON_PATH}")
    print(f"[OK] Total GPT messages counted: {total_gpt_msgs}")
    if TEST_LABELS_PATH:
        print(f"[OK] TEST labels loaded from: {TEST_LABELS_PATH} (ONLY_LABELS_IN_TEST={ONLY_LABELS_IN_TEST})")
    print(f"[OK] Saved CSV : {OUTPUT_CSV_PATH}")
    print(f"[OK] Saved JSON: {OUTPUT_JSON_PATH}")
    print("\nTop-10 labels by train_msg_freq:")
    for r in rows[:10]:
        print(f"  {r['label']:>12s} | msg_freq={r['train_msg_freq']:.6f} | msg_hits={r['train_msg_hits']} | occ_hits={r['train_occ_hits']}")


if __name__ == "__main__":
    main()
