import tempfile
from dataclasses import dataclass

import pandas as pd
import ray.data
from ray.data.extensions import TensorDtype
from pyarrow.fs import FileType
from hydra.core.config_store import ConfigStore

from x2r.configs import PrepareConfig
from x2r.io import get_global_filesystem
from .prepare import Prepare


@dataclass(kw_only=True)
class CIFAR10PrepareConfig(PrepareConfig):
    _target_: str = "x2r.datasets.prepare.CIFAR10Prepare"


cs = ConfigStore.instance()
cs.store(group="datasets/prepare", name="CIFAR10", node=CIFAR10PrepareConfig)


class CIFAR10Prepare(Prepare):
    def run(self) -> None:
        # TODO: rewrite

        from torchvision.datasets import CIFAR10

        fs = self.filesystem or get_global_filesystem()
        for split in ("train", "test"):
            status_path = str(self.output_dir / f"{split}.status")
            if fs.get_file_info(status_path).type is not FileType.NotFound:
                continue

            data_path = str(self.output_dir / split)
            if fs.get_file_info(data_path).type is not FileType.NotFound:
                fs.delete_dir_contents(data_path)

            with tempfile.TemporaryDirectory() as tmp_dir:
                dataset = CIFAR10(
                    root=tmp_dir if self.cache_dir is None else self.cache_dir,
                    train=split == "train",
                    download=True,
                )

            data, targets = dataset.data, dataset.targets

            df = pd.DataFrame({"image": list(data), "label": targets})
            df["image"] = df["image"].astype(TensorDtype(data.dtype, data.shape))

            ds = ray.data.from_pandas(df)

            ds = ds.repartition(8, shuffle=False)
            ds.write_parquet(
                data_path,
                filesystem=fs,
                compression="zstd",
            )

            with fs.open_output_stream(status_path) as stream:
                stream.write(b"ok")
