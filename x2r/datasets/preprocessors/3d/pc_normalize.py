from functools import partial
from typing import Dict, Optional, Union, Literal

import numpy as np
from ray.air.util.data_batch_conversion import BatchFormat
from ray.data.preprocessors import BatchMapper


def pc_normalize(
    batch: Dict[str, np.ndarray],
    pc_column: str,
    mean_column: str,
    std_column: str,
) -> Dict[str, np.ndarray]:
    outputs = batch.copy()
    pc: np.ndarray = batch[pc_column]
    pc_mean: np.ndarray = batch[mean_column]
    pc_std: np.ndarray = batch[std_column]

    if len(pc_mean.shape) == 1:
        pc_mean = pc_mean[:, None]

    if len(pc_std.shape) == 1:
        pc_std = pc_std[:, None]

    pc_mean = np.broadcast_to(pc_mean[:, None, :], pc.shape)
    pc_std = np.broadcast_to(pc_std[:, None, :], pc.shape)
    outputs[pc_column] = (pc - pc_mean) / pc_std
    return outputs


class PCNormalize(BatchMapper):
    def __init__(
        self,
        pc_column: str,
        mean_column: str,
        std_column: str,
        batch_size: Optional[Union[int, Literal["default"]]] = "default",
    ):
        super().__init__(
            partial(pc_normalize, pc_column=pc_column, mean_column=mean_column, std_column=std_column),
            batch_format=BatchFormat.NUMPY,
            batch_size=batch_size,
        )
