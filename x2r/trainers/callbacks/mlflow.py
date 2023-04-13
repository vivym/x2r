from dataclasses import dataclass
from typing import Optional, Dict, Union

from ray.air.integrations.mlflow import MLflowLoggerCallback
from hydra.core.config_store import ConfigStore


@dataclass(kw_only=True)
class MLflowLoggerCallbackConfig:
    _target_: str = "ray.air.integrations.mlflow.MLflowLoggerCallback"

    tracking_uri: Optional[str] = None
    registry_uri: Optional[str] = None
    experiment_name: Optional[str] = None
    tags: Optional[Dict[str, Union[str, int, float, bool]]] = None
    tracking_token: Optional[str] = None
    save_artifact: bool = False


cs = ConfigStore.instance()
cs.store(group="trainer/callbacks", name="MLflowLoggerCallback", node=MLflowLoggerCallbackConfig)
