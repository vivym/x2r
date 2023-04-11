from functools import partial
from typing import Dict, Optional, Union, Literal, List

import numpy as np
from ray.air.util.data_batch_conversion import BatchFormat
from ray.data.preprocessors import BatchMapper


def pc_random_downsample(
    batch: Union[np.ndarray, Dict[str, np.ndarray]],
    columns: List[str],
    num_points: int,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    def transform(pc: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng()
        return rng.choice(pc, num_points, axis=1, replace=False)

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


class PCRandomDownsample(BatchMapper):
    def __init__(
        self,
        columns: List[str],
        num_points: int,
        batch_size: Optional[Union[int, Literal["default"]]] = "default",
    ):
        super().__init__(
            partial(pc_random_downsample, columns=columns, num_points=num_points),
            batch_format=BatchFormat.NUMPY,
            batch_size=batch_size,
        )
