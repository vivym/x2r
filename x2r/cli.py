from typing import Tuple, Optional

from ray.data import Dataset as RayDataset
from ray.data.preprocessors import Chain
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from .configs import Config, TaskType, DatasetsConfig
from .datasets import Dataset
from .datasets.prepare import Prepare
from .io import set_global_filesystem


def parse_datasets(
    datasets: DatasetsConfig
) -> Tuple[Optional[Prepare], Optional[RayDataset], Optional[RayDataset], Optional[RayDataset]]:
    if datasets.prepare is not None:
        prepare = instantiate(datasets.prepare)
        prepare.run()

    if datasets.preprocessors is not None:
        preprocessors = [instantiate(preprocessor) for preprocessor in datasets.preprocessors]
        preprocessor = Chain(*preprocessors)
    else:
        preprocessor = None

    if datasets.train is not None:
        train_dataset: Dataset = instantiate(datasets.train)
        train_ray_dataset = train_dataset.get_ray_dataset()

        if preprocessor is None:
            preprocessor = train_dataset.get_preprocessors()
    else:
        train_ray_dataset = None

    if datasets.val is not None:
        val_dataset: Dataset = instantiate(datasets.val)
        val_ray_dataset = val_dataset.get_ray_dataset()
    else:
        val_ray_dataset = None

    if datasets.test is not None:
        test_dataset: Dataset = instantiate(datasets.test)
        test_ray_dataset = test_dataset.get_ray_dataset()
    else:
        test_ray_dataset = None

    return preprocessor, train_ray_dataset, val_ray_dataset, test_ray_dataset


def main(cfg: DictConfig):
    cfg: Config = OmegaConf.to_object(cfg)

    set_global_filesystem(cfg.filesystem)

    if cfg.task == TaskType.TRAIN:
        preprocessor, train_dataset, val_dataset, test_dataset = parse_datasets(cfg.datasets)

        model_factory = instantiate(cfg.model, _partial_=True)

        optimizer_factory = instantiate(cfg.optimizer, _partial_=True)

        if cfg.lr_scheduler is None:
            lr_scheduler_factory = lambda _: None
        else:
            lr_scheduler_factory = instantiate(cfg.lr_scheduler, _partial_=True)

        overrides = {
            "model_factory": model_factory,
            "optimizer_factory": optimizer_factory,
            "lr_scheduler_factory": lr_scheduler_factory,
            "datasets": {},
            "preprocessor": preprocessor,
        }

        if train_dataset is not None:
            overrides["datasets"]["train"] = train_dataset

        if val_dataset is not None:
            overrides["datasets"]["val"] = val_dataset

        if test_dataset is not None:
            overrides["datasets"]["test"] = test_dataset

        trainer = instantiate(cfg.trainer, **overrides)
        trainer.fit()
