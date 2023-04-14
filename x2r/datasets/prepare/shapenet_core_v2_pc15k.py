import json
import tempfile
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import gdown
import numpy as np
import pandas as pd
import ray.data
from ray.data.extensions import TensorDtype
from rich.progress import track
from pyarrow.fs import FileType
from hydra.core.config_store import ConfigStore

from x2r.configs import PrepareConfig
from x2r.io import get_global_filesystem
from .prepare import Prepare

synset_id_to_category = {
    "02691156": "airplane", "02773838": "bag", "02801938": "basket",
    "02808440": "bathtub", "02818832": "bed", "02828884": "bench",
    "02876657": "bottle", "02880940": "bowl", "02924116": "bus",
    "02933112": "cabinet", "02747177": "can", "02942699": "camera",
    "02954340": "cap", "02958343": "car", "03001627": "chair",
    "03046257": "clock", "03207941": "dishwasher", "03211117": "monitor",
    "04379243": "table", "04401088": "telephone", "02946921": "tin_can",
    "04460130": "tower", "04468005": "train", "03085013": "keyboard",
    "03261776": "earphone", "03325088": "faucet", "03337140": "file",
    "03467517": "guitar", "03513137": "helmet", "03593526": "jar",
    "03624134": "knife", "03636649": "lamp", "03642806": "laptop",
    "03691459": "speaker", "03710193": "mailbox", "03759954": "microphone",
    "03761084": "microwave", "03790512": "motorcycle", "03797390": "mug",
    "03928116": "piano", "03938244": "pillow", "03948459": "pistol",
    "03991062": "pot", "04004475": "printer", "04074963": "remote_control",
    "04090263": "rifle", "04099429": "rocket", "04225987": "skateboard",
    "04256520": "sofa", "04330267": "stove", "04530566": "vessel",
    "04554684": "washer", "02992529": "cellphone",
    "02843684": "birdhouse", "02871439": "bookshelf",
    # "02858304": "boat", no boat in our dataset, merged into vessels
    # "02834778": "bicycle", not in our taxonomy
}
categories = sorted(synset_id_to_category.values())
category_to_category_id = {v: k for k, v in enumerate(categories)}
category_to_synset_id = {v: k for k, v in synset_id_to_category.items()}


@dataclass(kw_only=True)
class ShapeNetCoreV2PC15kPrepareConfig(PrepareConfig):
    _target_: str = "x2r.datasets.prepare.ShapeNetCoreV2PC15kPrepare"


cs = ConfigStore.instance()
cs.store(group="datasets/prepare", name="ShapeNetCoreV2PC15k", node=ShapeNetCoreV2PC15kPrepareConfig)


class ShapeNetCoreV2PC15kPrepare(Prepare):
    def run(self) -> None:
        splits = ["train", "val", "test"]

        if self.cache_dir is not None:
            cache_dir = self.cache_dir
        else:
            cache_dir = Path(tempfile.mkdtemp())

        fs = self.filesystem or get_global_filesystem()

        pc_infos_path = str(self.output_dir / f"pc_infos.json")
        stats_infos_path = str(self.output_dir / f"stats_infos.json")

        if fs.get_file_info(pc_infos_path).type is FileType.NotFound or fs.get_file_info(
            stats_infos_path
        ).type is FileType.NotFound:
            gdown.cached_download(
                "https://drive.google.com/u/0/uc?id=1s2NA1unoLScx0GYcZfQ7ot3XkwoH_0l6",
                str(cache_dir / "ShapeNetCore.v2.PC15k.zip"),
                quiet=False,
                md5="22660aab28f604a62ca6c4d23811200e",
                postprocess=gdown.extractall,
            )
            pc_infos = {split: [] for split in splits}
            stats_infos = {synset_id: [] for synset_id in synset_id_to_category.keys()}

            for synset_id_path in track(list((cache_dir / "ShapeNetCore.v2.PC15k").glob("*"))):
                synset_id = synset_id_path.name
                if not synset_id_path.is_dir() or synset_id not in synset_id_to_category:
                    continue
                category = synset_id_to_category[synset_id]

                pcs = []

                for split in splits:
                    for pc_path in (synset_id_path / split).glob("*.npy"):
                        pc = np.load(pc_path).astype(np.float32)
                        assert pc.shape[0] == 15000
                        pcs.append(pc)

                        pc_infos[split].append({
                            "category": category,
                            "category_id": category_to_category_id[category],
                            "synset_id": synset_id,
                            "pc_path": str(pc_path.relative_to(cache_dir)),
                        })

                pcs = np.concatenate(pcs, axis=0)
                pc_mean = pcs.mean(0).tolist()
                pc_std = pcs.std(0).tolist()
                pc_std_all = pcs.reshape(-1).std(0).item()

                stats_infos[synset_id] = {
                    "mean": pc_mean,
                    "std": pc_std,
                    "std_all": pc_std_all,
                }

            with fs.open_output_stream(pc_infos_path) as stream:
                stream.write(json.dumps(pc_infos).encode("utf-8"))

            with fs.open_output_stream(stats_infos_path) as stream:
                stream.write(json.dumps(stats_infos).encode("utf-8"))

        with fs.open_input_stream(pc_infos_path) as stream:
            pc_infos = json.load(stream)

        with fs.open_input_stream(stats_infos_path) as stream:
            stats_infos = json.load(stream)

        def load_pc(data: Dict[str, Any]):
            synset_id =  data["synset_id"]
            pc_mean = stats_infos[synset_id]["mean"]
            pc_std = stats_infos[synset_id]["std"]
            pc_std_all = stats_infos[synset_id]["std_all"]

            pc_mean = np.asarray(pc_mean, dtype=np.float32)
            pc_std = np.asarray(pc_std, dtype=np.float32)
            pc_std_all = np.asarray(pc_std_all, dtype=np.float32)

            pc_path = cache_dir / data["pc_path"]
            pc = np.load(pc_path).astype(np.float32)
            return dict(**data, pc=pc, pc_mean=pc_mean, pc_std=pc_std, pc_std_all=pc_std_all)

        for split in splits:
            status_path = str(self.output_dir / f"{split}.status")
            if fs.get_file_info(status_path).type is not FileType.NotFound:
                continue

            data_path = str(self.output_dir / split)
            if fs.get_file_info(data_path).type is not FileType.NotFound:
                fs.delete_dir_contents(data_path)

            ds = ray.data.from_items(pc_infos[split])
            ds = ds.repartition(64, shuffle=False)
            ds = ds.map(load_pc)
            ds.write_parquet(
                data_path,
                filesystem=fs,
                compression="zstd",
            )

            with fs.open_output_stream(status_path) as stream:
                stream.write(b"ok")

        if self.cache_dir is None:
            shutil.rmtree(cache_dir)
