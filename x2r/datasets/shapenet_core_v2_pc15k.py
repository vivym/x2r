from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from ray.data.datasource import DefaultParquetMetadataProvider, ParquetMetadataProvider
from hydra.core.config_store import ConfigStore

from .parquet_dataset import ParquetDataset, ParquetDatasetConfig


@dataclass(kw_only=True)
class ShapeNetCoreV2PC15kConfig(ParquetDatasetConfig):
    _target_: str = "x2r.datasets.ShapeNetCoreV2PC15kDataseet"

    categories: Optional[List[str]] = None


cs = ConfigStore.instance()

for split in ("train", "val", "test"):
    cs.store(group=f"datasets/{split}", name="ShapeNetCoreV2PC15k", node=ShapeNetCoreV2PC15kConfig)


class ShapeNetCoreV2PC15kDataseet(ParquetDataset):
    def __init__(
        self,
        path: Optional[str] = None,
        paths: Optional[List[str]] = None,
        filesystem: Optional["pa.fs.FileSystem"] = None,
        columns: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        parallelism: int = -1,
        ray_remote_args: Optional[Dict[str, Any]] = None,
        tensor_column_schema: Optional[Dict[str, Tuple[np.dtype, Tuple[int, ...]]]] = None,
        meta_provider: ParquetMetadataProvider = DefaultParquetMetadataProvider(),
        arrow_parquet_args: Optional[Dict[str, Any]] = None,
    ):
        drop_category = False

        if categories is not None:
            if arrow_parquet_args is None:
                arrow_parquet_args = {}
            arrow_parquet_args["filter"] = pc.field("category").isin(categories)

            if "category" not in columns:
                columns = columns + ["category"]
                drop_category = True

        super().__init__(
            path=path,
            paths=paths,
            filesystem=filesystem,
            columns=columns,
            parallelism=parallelism,
            ray_remote_args=ray_remote_args,
            tensor_column_schema=tensor_column_schema,
            meta_provider=meta_provider,
            arrow_parquet_args=arrow_parquet_args,
        )

        if drop_category:
            self.dataset = self.dataset.drop_columns("category")
