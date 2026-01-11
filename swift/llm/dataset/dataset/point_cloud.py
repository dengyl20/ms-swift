# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import json
import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from datasets import Dataset as HfDataset
from datasets import Features
from datasets import IterableDataset as HfIterableDataset
from datasets import Sequence, Value
from torch.utils.data import Dataset

from swift.utils import get_logger
from ..preprocessor import MessagesPreprocessor, RowPreprocessor
from ..register import DatasetMeta, register_dataset

logger = get_logger()


class ObjectPointCloudDataset(Dataset):
    """Point cloud dataset for instruction tuning."""

    def __init__(
        self,
        data_path: str,
        anno_path: str,
        pointnum: int = 8192,
        split: str = 'train',
        conversation_types: Optional[Iterable[str]] = None,
        use_color: bool = True,
        normalize_pc: bool = True,
        return_torch: bool = True,
        split_train_val: bool = False,
        split_ratio: float = 0.9,
        data_debug_num: int = 0,
        point_indicator: str = '<point>',
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.anno_path = anno_path
        self.pointnum = int(pointnum)
        self.split = split
        self.use_color = bool(use_color)
        self.normalize_pc = bool(normalize_pc)
        self.return_torch = bool(return_torch)
        self.split_train_val = bool(split_train_val)
        self.split_ratio = float(split_ratio)
        self.data_debug_num = int(data_debug_num)
        self.point_indicator = point_indicator

        logger.info(f'Loading anno file from {anno_path}.')
        with open(anno_path, 'r') as f:
            self.list_data_dict = json.load(f)

        logger.info(f'Before filtering, dataset size: {len(self.list_data_dict)}')

        corrupted_color_ids = {
            '6760e543e1d645d5aaacd3803bcae524',
            'b91c0711149d460a8004f9c06d3b7f38',
        }
        if self.use_color:
            self.list_data_dict = [
                d for d in self.list_data_dict if d.get('object_id') not in corrupted_color_ids
            ]

        if conversation_types is not None:
            if isinstance(conversation_types, (list, tuple, set)):
                conversation_types = tuple(conversation_types)
            else:
                conversation_types = (conversation_types,)
            self.list_data_dict = [
                d for d in self.list_data_dict
                if d.get('conversation_type', 'simple_description') in conversation_types
            ]
            logger.info(f'Using conversation_types: {conversation_types}')
        else:
            logger.info('conversation_types=None => keep all conversation types.')

        logger.info(f'After filtering, dataset size: {len(self.list_data_dict)}')

        if self.data_debug_num > 0:
            self.list_data_dict = self.list_data_dict[:self.data_debug_num]
            object_ids = ' '.join([d['object_id'] for d in self.list_data_dict])
            logger.info(f'Debug mode, using object_ids: {object_ids}')

        if self.split_train_val:
            n_total = len(self.list_data_dict)
            n_train = int(self.split_ratio * n_total)
            if self.split == 'train':
                self.list_data_dict = self.list_data_dict[:n_train]
                logger.info(f'Train set size: {len(self.list_data_dict)}')
            else:
                self.list_data_dict = self.list_data_dict[n_train:]
                logger.info(f'Val set size: {len(self.list_data_dict)}')

    def __len__(self) -> int:
        return len(self.list_data_dict)

    def _pointcloud_file(self, object_id: str) -> str:
        return os.path.join(self.data_path, f'{object_id}_{self.pointnum}.npy')

    def _load_point_cloud(self, object_id: str) -> np.ndarray:
        pc_path = self._pointcloud_file(object_id)
        if not os.path.isfile(pc_path):
            raise FileNotFoundError(f'Point cloud file not found: {pc_path}')
        point_cloud = np.load(pc_path)
        if not self.use_color:
            point_cloud = point_cloud[:, :3]
        return point_cloud

    @staticmethod
    def pc_norm(pc: np.ndarray) -> np.ndarray:
        pc = pc.astype(np.float32, copy=False)
        xyz = pc[:, :3]
        other = pc[:, 3:] if pc.shape[1] > 3 else None
        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        m = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
        if m < 1e-12:
            m = 1e-12
        xyz = xyz / m
        if other is None:
            return xyz
        return np.concatenate([xyz, other], axis=1)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.list_data_dict[index]
        object_id = item['object_id']
        conversations = copy.deepcopy(item.get('conversations', []))
        point_cloud = self._load_point_cloud(object_id)
        if self.normalize_pc:
            point_cloud = self.pc_norm(point_cloud)
        point_cloud = point_cloud.astype(np.float32, copy=False)
        if self.return_torch:
            point_cloud = torch.from_numpy(point_cloud)
        return {
            'object_id': object_id,
            'conversations': conversations,
            'point_clouds': point_cloud,
        }


def _parse_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.lower() in {'1', 'true', 'yes', 'y', 't'}


def _parse_pointcloud_kwargs(extra_args: List[str]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    for item in extra_args:
        if '=' not in item:
            continue
        key, value = item.split('=', 1)
        kwargs[key] = value
    return kwargs


def _iter_pointcloud_dataset(
    dataset: ObjectPointCloudDataset,
    *,
    row_preprocessor: Optional['StreamingRowPreprocessor'] = None,
) -> Iterable[Dict[str, Any]]:
    for idx in range(len(dataset)):
        row = dataset[idx]
        if row_preprocessor is not None:
            row = row_preprocessor(row)
            if row is None:
                continue
        yield row


class StreamingRowPreprocessor:

    def __init__(self) -> None:
        self.messages_preprocessor = MessagesPreprocessor(
            role_key='from',
            content_key='value',
            user_role='human',
            assistant_role='gpt',
        )

    def __call__(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        item = {
            'object_id': row['object_id'],
            'messages': copy.deepcopy(row.get('conversations', [])),
            'points': row['point_clouds'],
        }
        item = self.messages_preprocessor.preprocess(item)
        if item is None:
            return None
        RowPreprocessor._check_messages(item)
        RowPreprocessor._cast_mm_data(item)
        return item


def _build_streaming_features(pointnum: int, use_color: bool) -> Features:
    channels = 6 if use_color else 3
    return Features({
        'object_id': Value('string'),
        # 关键修改：用 python list 表示 list-of-struct
        'messages': [{
            'role': Value('string'),
            'content': Value('string'),
        }],
        'points': Sequence(Sequence(Value('float32'), length=channels), length=pointnum),
    })


def load_pointcloud_dataset(
    dataset_syntax,
    dataset_meta: DatasetMeta,
    *,
    num_proc: int = 1,
    load_from_cache_file: bool = True,
    streaming: bool = True,
    strict: bool = False,
    columns: Optional[Dict[str, str]] = None,
    remove_unused_columns: bool = True,
    **kwargs,
) -> HfDataset:
    options = _parse_pointcloud_kwargs(dataset_syntax.subsets)
    data_path = options.get('data_path') or os.getenv('POINT_CLOUD_DATA_PATH')
    anno_path = options.get('anno_path') or os.getenv('POINT_CLOUD_ANNO_PATH')
    if not data_path or not anno_path:
        raise ValueError(
            'Point cloud dataset requires data_path and anno_path. '
            'Provide them via --dataset pointllm_point_cloud:data_path=...:anno_path=... '
            'or set POINT_CLOUD_DATA_PATH and POINT_CLOUD_ANNO_PATH env vars.')

    conversation_types = options.get('conversation_types')
    if conversation_types and ',' in conversation_types:
        conversation_types = [t for t in conversation_types.split(',') if t]
    use_color = _parse_bool(options.get('use_color'), True)
    pointnum = int(options.get('pointnum', 8192))
    dataset = ObjectPointCloudDataset(
        data_path=data_path,
        anno_path=anno_path,
        pointnum=pointnum,
        split=options.get('split', 'train'),
        conversation_types=conversation_types,
        use_color=use_color,
        normalize_pc=_parse_bool(options.get('normalize_pc'), True),
        return_torch=False,
        split_train_val=_parse_bool(options.get('split_train_val'), False),
        split_ratio=float(options.get('split_ratio', 0.9)),
        data_debug_num=int(options.get('data_debug_num', 0)),
    )
    if streaming:
        features = _build_streaming_features(pointnum, use_color)
        dataset = HfIterableDataset.from_generator(
            _iter_pointcloud_dataset,
            gen_kwargs={
                'dataset': dataset,
                'row_preprocessor': StreamingRowPreprocessor(),
            },
            features=features)
        if columns:
            dataset = RowPreprocessor.safe_rename_columns(dataset, columns)
        if remove_unused_columns:
            dataset = RowPreprocessor.remove_useless_columns(dataset)
        return dataset

    dataset = HfDataset.from_generator(_iter_pointcloud_dataset, gen_kwargs={'dataset': dataset})

    if columns:
        dataset = RowPreprocessor.safe_rename_columns(dataset, columns)
    dataset = dataset_meta.preprocess_func(
        dataset, num_proc=num_proc, load_from_cache_file=load_from_cache_file, strict=strict)
    if remove_unused_columns:
        dataset = RowPreprocessor.remove_useless_columns(dataset)
    return dataset


register_dataset(
    DatasetMeta(
        dataset_name='pointllm_point_cloud',
        preprocess_func=MessagesPreprocessor(
            role_key='from',
            content_key='value',
            user_role='human',
            assistant_role='gpt',
            columns={
                'conversations': 'messages',
                'point_clouds': 'points',
            },
        ),
        load_function=load_pointcloud_dataset,
        huge_dataset=True,
        tags=['chat', 'multi-modal', 'point-cloud'],
        help='Point cloud instruction tuning dataset using <point> placeholders.',
    ))
