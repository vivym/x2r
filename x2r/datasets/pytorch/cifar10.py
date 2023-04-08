from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from PIL import Image
from ray.data import from_torch, Dataset
from ray.data.preprocessor import Preprocessor
from ray.data.preprocessors import TorchVisionPreprocessor
from hydra.core.config_store import ConfigStore
from torchvision.datasets import CIFAR10 as _CIFAR10
from torchvision import transforms as T

from x2r.configs import DatasetConfigBase
from x2r.datasets import DatasetBase


@dataclass(kw_only=True)
class CIFAR10DatasetConfig(DatasetConfigBase):
    _target_: str = "x2r.datasets.pytorch.CIFAR10"

    root_path: str


cs = ConfigStore.instance()
cs.store(group="dataset", name="CIFAR10", node=CIFAR10DatasetConfig)


def convert_batch_to_numpy(batch: Tuple[Image.Image, int]) -> Dict[str, np.ndarray]:
    images = np.stack([np.array(image) for image, _ in batch])
    labels = np.array([label for _, label in batch])
    return {"image": images, "label": labels}


class CIFAR10(DatasetBase):
    def __init__(self, root_path: str):
        super().__init__()

        self.root_path = root_path

        # TODO: lazy load?
        train_dataset = _CIFAR10("data", download=True, train=True)
        test_dataset = _CIFAR10("data", download=True, train=False)

        self.train_dataset: Dataset = from_torch(train_dataset).map_batches(
            convert_batch_to_numpy
        )
        self.test_dataset: Dataset = from_torch(test_dataset).map_batches(
            convert_batch_to_numpy
        )

        transform = T.Compose(
            [T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        self.preprocessor = TorchVisionPreprocessor(
            columns=["image"], transform=transform
        )

    def get_preprocessor(self) -> Preprocessor:
        return self.preprocessor

    def get_train_dataset(self) -> Dataset:
        return self.train_dataset

    def get_val_dataset(self) -> Dataset:
        return self.test_dataset

    def get_test_dataset(self) -> Dataset:
        return self.test_dataset
