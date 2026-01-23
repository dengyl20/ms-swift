# pc_constants.py
from __future__ import annotations

POINT_TOKEN = "<pointcloud>"

# 复用你推理脚本里的 system prompt（注意：不出现字面 "<point>"）
DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of understanding text inputs "
    "and generating helpful responses.\n\n"
    "Task setting:\n"
    "- You will answer questions about an object represented by a 3D point cloud.\n"
    "- In some requests, the user message will contain a section named '3D_POINT_CLOUD_EMBEDDING'. "
    "The tokens in that section are placeholders whose embeddings are injected at inference/training time to carry "
    "semantic information about the 3D object.\n\n"
    "Instructions:\n"
    "- Use the '3D_POINT_CLOUD_EMBEDDING' section as object context to answer the question.\n"
    "- If the embedding section is absent, answer based only on the text question and be explicit about uncertainty.\n"
    "- Output only the final answer text (no role labels such as user/assistant, no extra dialogue markers).\n"
)

# ===== 环境变量（你运行前需要 export）=====
ENV_FEATURE_INFO_YAML = "POINT_FEATURE_DATASET_INFO_YAML"
ENV_CONV_JSON_PATH = "POINT_CONV_JSON_PATH"
ENV_AE_CKPT_PATH = "POINT_AE_CKPT_PATH"

# 可选：控制注入 token cap（默认 128，与推理脚本一致）
ENV_MAX_INJECT_TOKENS = "POINT_MAX_INJECT_TOKENS"
DEFAULT_MAX_INJECT_TOKENS = 24

# 可选：dataset require_valid
ENV_REQUIRE_VALID = "POINT_REQUIRE_VALID"
DEFAULT_REQUIRE_VALID = True
