from typing import Callable, Optional

import torch
import torch.nn as nn
from ray.air import session
from ray.train.torch import prepare_model, get_device
from rich.progress import Progress


def default_training_loop_per_worker(
    model_factory: Callable[[], nn.Module],
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
    lr_scheduler_factory: Callable[[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler]],
    batch_size: int,
    max_epochs: int = 1000,
    max_steps: Optional[int] = None,
    val_every_n_epochs: int = 1,
    val_every_n_steps: Optional[int] = None,
    lr_scheduler_on_step: bool = False,
):
    device = get_device()

    model = prepare_model(model_factory())

    optimizer = optimizer_factory(model.parameters())
    lr_scheduler = lr_scheduler_factory(optimizer)

    train_dataset_shard = session.get_dataset_shard("train")
    val_dataset_shard = session.get_dataset_shard("val")

    done = False
    current_steps = 0

    has_validation_step = hasattr(model, "validation_step")

    progress = Progress()
    progress.start()

    if max_steps is None:
        epoch_task = progress.add_task("[red]Epoch", total=max_epochs)
        num_steps_per_epoch = None
        step_task = progress.add_task("[green]Step", total=None)
    else:
        step_task = progress.add_task("[green]Step", total=max_steps)

    try:
        for current_epoch in range(max_epochs):
            train_dataset_iter = train_dataset_shard.iter_torch_batches(
                batch_size=batch_size, device=device
            )

            model.train()

            if max_steps is None:
                progress.reset(epoch_task, total=num_steps_per_epoch)

            num_steps_per_epoch = 0

            for i, batch in enumerate(train_dataset_iter):
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                loss: torch.Tensor = model.training_step(batch, i)

                if isinstance(loss, torch.Tensor):
                    loss.backward()
                    optimizer.step()

                if lr_scheduler is not None and lr_scheduler_on_step:
                    lr_scheduler.step()

                progress.update(step_task, advance=1)

                current_steps += 1
                num_steps_per_epoch += 1
                if max_steps is not None and current_steps >= max_steps:
                    done = True
                    break

            if has_validation_step and val_dataset_shard is not None and current_epoch % val_every_n_epochs == 0:
                val_dataset_iter = val_dataset_shard.iter_torch_batches(
                    batch_size=batch_size, device=device
                )

                model.eval()

                with torch.no_grad():
                    for i, batch in enumerate(val_dataset_iter):
                        model.validation_step(batch, i)

            if lr_scheduler is not None and not lr_scheduler_on_step:
                lr_scheduler.step()

            if max_steps is None:
                progress.update(epoch_task, advance=1)

            if done:
                break
    finally:
        progress.stop()
