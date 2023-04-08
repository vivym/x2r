from typing import Any

import torch.nn as nn



class BaseModel(nn.Module):
    def training_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError("training_step not implemented")

    def validation_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError("validation_step not implemented")

    def test_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError("test_step not implemented")
