from functools import partial
from typing import Dict, Optional, Union, Literal, List

import numpy as np
from ray.air.util.data_batch_conversion import BatchFormat
from ray.data.preprocessors import BatchMapper


def to_float(
    batch: Union[np.ndarray, Dict[str, np.ndarray]],
    columns: List[str],
    norm_factor: float,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    def transform(data: np.ndarray) -> np.ndarray:
        return data.astype(np.float32) / norm_factor

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


class Uint8ToFloat(BatchMapper):
    def __init__(
        self,
        columns: List[str],
        batch_size: Optional[Union[int, Literal["default"]]] = "default",
    ):
        super().__init__(
            partial(to_float, columns=columns, norm_factor=255.),
            batch_format=BatchFormat.NUMPY,
            batch_size=batch_size,
        )
