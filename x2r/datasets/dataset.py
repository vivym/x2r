from typing import Optional, List

from ray.data import Dataset as RayDataset
from ray.data.preprocessor import Preprocessor


class Dataset:
    def __init__(self, dataset: RayDataset) -> None:
        # TODO: support windowing config

        self.dataset = dataset

    def get_ray_dataset(self) -> RayDataset:
        return self.dataset

    def get_preprocessors(self) -> Optional[List[Preprocessor]]:
        return None
