# modelnet40_feature_dataset.py
from __future__ import annotations

import os
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


# ============================================================
# 全局配置（按需直接改这里；不使用 argparse）
# ============================================================
FEATURE_PT_PATH = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/PointLLM/PointLLM/modelnet40_gray_color.pt"

# 是否只保留 valid==1 的样本（默认 False：小测试集一般不需要过滤；但你可打开）
REQUIRE_VALID = True


class ModelNet40PointTokenDataset(Dataset):
    """
    读取 extract_modelnet40_features.py 生成的 .pt 特征文件。
    __getitem__ 返回：
      - point_tokens: (T,D) torch.Tensor
      - object_labels: str
    """

    def __init__(self, feature_pt_path: str, require_valid: bool = False):
        super().__init__()
        self.feature_pt_path = feature_pt_path
        self.require_valid = bool(require_valid)

        if not os.path.isfile(feature_pt_path):
            raise FileNotFoundError(feature_pt_path)

        data = torch.load(feature_pt_path, map_location="cpu")

        self.point_tokens = data["point_tokens"]          # (N,T,D)
        self.object_labels = data["object_labels"]        # list[str]
        self.valid = data.get("valid", None)              # (N,) uint8 or None

        if not isinstance(self.object_labels, list):
            raise ValueError("object_labels must be a list[str]")

        if self.point_tokens.ndim != 3:
            raise ValueError(f"point_tokens must be (N,T,D), got shape={tuple(self.point_tokens.shape)}")

        if len(self.object_labels) != self.point_tokens.shape[0]:
            raise ValueError("len(object_labels) must match point_tokens.shape[0]")

        self.n = int(self.point_tokens.shape[0])

        # 可选：构建有效索引
        if self.require_valid:
            if self.valid is None:
                raise ValueError("require_valid=True but feature file has no 'valid' field")
            v = self.valid.to(torch.uint8).view(-1)
            self.indices = torch.nonzero(v > 0, as_tuple=False).view(-1).tolist()
        else:
            self.indices = list(range(self.n))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = self.indices[idx]
        pt = self.point_tokens[real_idx]   # (T,D)
        lab = self.object_labels[real_idx]
        return {
            "point_tokens": pt,
            "object_labels": lab,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    point_tokens = torch.stack([b["point_tokens"] for b in batch], dim=0)  # (B,T,D)
    object_labels = [b["object_labels"] for b in batch]
    return {
        "point_tokens": point_tokens,
        "object_labels": object_labels,
    }


def _smoke_test() -> None:
    ds = ModelNet40PointTokenDataset(FEATURE_PT_PATH, require_valid=REQUIRE_VALID)
    x0 = ds[0]
    print(f"[dataset] len={len(ds)}")
    print(f"[dataset] point_tokens[0].shape={tuple(x0['point_tokens'].shape)}, dtype={x0['point_tokens'].dtype}")
    print(f"[dataset] object_labels[0]={x0['object_labels']}")


if __name__ == "__main__":
    _smoke_test()
