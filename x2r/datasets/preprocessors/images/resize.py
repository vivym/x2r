from functools import partial
from typing import Dict, Union, List, Tuple

import cv2
import numpy as np
from ray.air.util.data_batch_conversion import BatchFormat
from ray.data.preprocessors import BatchMapper


def resize(
    batch: Union[np.ndarray, Dict[str, np.ndarray]],
    columns: List[str],
    size: Tuple[int, int],
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    def transform(images: np.ndarray) -> np.ndarray:
        resized_images = []
        for image in images:
            image = cv2.resize(image, size)
            resized_images.append(image)
        return np.stack(resized_images, axis=0)

    if isinstance(batch, dict):
        outputs = {}
        for key, data in batch.items():
            if key in columns:
                outputs[key] = transform(data)
            else:
                outputs[key] = data
    else:
        outputs = transform(batch)

    return outputs


class Resize(BatchMapper):
    def __init__(
        self,
        columns: List[str],
        size: Tuple[int, int],
        batch_size: int = 1,
    ):
        super().__init__(
            partial(resize, columns=columns, size=size),
            batch_format=BatchFormat.NUMPY,
            batch_size=batch_size,
        )
