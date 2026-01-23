# pc_utils.py
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple


def strip_all_point_placeholders(text: str, placeholder: str = "<point>") -> str:
    """
    删除文本里所有 <point> 字面串（用于从原始 human prompt 提取“纯问题”）。
    """
    lines: List[str] = []
    for line in str(text).splitlines():
        line2 = line.replace(placeholder, " ")
        line2 = " ".join(line2.split())
        if line2.strip() != "":
            lines.append(line2)
    return "\n".join(lines).strip()


def build_user_prompt_with_points(question_text: str, k: int, placeholder: str = "<point>") -> str:
    """
    与你推理脚本一致的结构化 prompt。
    """
    k = max(1, int(k))
    q = str(question_text).strip()
    if q == "":
        q = "Describe the object represented by the 3D point cloud."
    point_block = " ".join([placeholder] * k)
    return (
        "3D_POINT_CLOUD_EMBEDDING:\n"
        f"{point_block}\n\n"
        "QUESTION:\n"
        f"{q}\n\n"
        "ANSWER:"
    )


def extract_first_round(conv_list: List[Dict[str, Any]]) -> Optional[Tuple[str, str]]:
    """
    conv_list: [{"from":"human","value":"..."}, {"from":"gpt","value":"..."}, ...]
    只取第一轮：第一个 human + 其后的第一个 gpt
    """
    if not isinstance(conv_list, list) or len(conv_list) == 0:
        return None

    human_text = None
    gpt_text = None

    human_idx = None
    for i, msg in enumerate(conv_list):
        if msg.get("from") == "human":
            human_text = msg.get("value", "")
            human_idx = i
            break

    if human_idx is None:
        return None

    for j in range(human_idx + 1, len(conv_list)):
        if conv_list[j].get("from") == "gpt":
            gpt_text = conv_list[j].get("value", "")
            break

    if human_text is None or gpt_text is None:
        return None
    return str(human_text), str(gpt_text)


def load_conversation_map(conv_json_path: str) -> Dict[str, Dict[str, str]]:
    """
    把大 JSON 里的 object_id -> {"human":..., "gpt":...} 全量读入内存。
    优先 ijson 流式；否则 fallback json.load（可能吃内存）。
    """
    out: Dict[str, Dict[str, str]] = {}

    # 优先 ijson
    try:
        import ijson  # type: ignore

        with open(conv_json_path, "rb") as f:
            for item in ijson.items(f, "item"):
                obj_id = item.get("object_id", None)
                if obj_id is None:
                    continue
                pair = extract_first_round(item.get("conversations", []))
                if pair is None:
                    continue
                human, gpt = pair
                out[str(obj_id)] = {"human": human, "gpt": gpt}
        return out
    except Exception:
        pass

    # fallback：整文件 load
    with open(conv_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        obj_id = item.get("object_id", None)
        if obj_id is None:
            continue
        pair = extract_first_round(item.get("conversations", []))
        if pair is None:
            continue
        human, gpt = pair
        out[str(obj_id)] = {"human": human, "gpt": gpt}
    return out
