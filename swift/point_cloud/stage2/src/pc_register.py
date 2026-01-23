# register.py
from __future__ import annotations

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from swift.point_cloud.stage2.src.pc_dataset import register_pointcloud_dataset
from swift.point_cloud.stage2.src.pc_model import register_qwen3_omni_point_model
from swift.point_cloud.stage2.src.pc_template import register_qwen3_omni_point_template


def register_all():
    # 顺序无强依赖，但建议：template + dataset + model
    register_qwen3_omni_point_template(exists_ok=True)
    register_pointcloud_dataset(exists_ok=True)
    register_qwen3_omni_point_model(exists_ok=True)


register_all()
