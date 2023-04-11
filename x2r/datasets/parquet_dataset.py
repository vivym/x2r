from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import ray.data
import pyarrow
from ray.data.datasource import DefaultParquetMetadataProvider, ParquetMetadataProvider
from hydra.core.config_store import ConfigStore

from x2r.configs import DatasetConfig
from x2r.io import get_global_filesystem
from .dataset import Dataset


@dataclass(kw_only=True)
class ParquetDatasetConfig(DatasetConfig):
    _target_: str = "x2r.datasets.ParquetDataset"

    path: Optional[str] = None
    paths: Optional[List[str]] = None
    filesystem: Optional[Dict[str, Any]] = None
    columns: Optional[List[str]] = None
    parallelism: int = -1
    ray_remote_args: Optional[Dict[str, Any]] = None
    arrow_parquet_args: Optional[Dict[str, Any]] = None


cs = ConfigStore.instance()

for split in ("train", "val", "test"):
    cs.store(group=f"datasets/{split}", name="ParquetDataset", node=ParquetDatasetConfig)


class ParquetDataset(Dataset):
    def __init__(
        self,
        path: Optional[str] = None,
        paths: Optional[List[str]] = None,
        filesystem: Optional["pyarrow.fs.FileSystem"] = None,
        columns: Optional[List[str]] = None,
        parallelism: int = -1,
        ray_remote_args: Optional[Dict[str, Any]] = None,
        tensor_column_schema: Optional[Dict[str, Tuple[np.dtype, Tuple[int, ...]]]] = None,
        meta_provider: ParquetMetadataProvider = DefaultParquetMetadataProvider(),
        arrow_parquet_args: Optional[Dict[str, Any]] = None,
    ):
        if paths is None:
            paths = []

        if path is not None:
            paths = [path] + list(paths)

        if filesystem is None:
            filesystem = get_global_filesystem()

        if arrow_parquet_args is None:
            arrow_parquet_args = {}

        dataset = ray.data.read_parquet(
            paths,
            filesystem=filesystem,
            columns=list(columns),
            parallelism=parallelism,
            ray_remote_args=ray_remote_args,
            tensor_column_schema=tensor_column_schema,
            meta_provider=meta_provider,
            **arrow_parquet_args,
        )
        super().__init__(dataset)
