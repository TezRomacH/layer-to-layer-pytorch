from typing import List

import copy

import numpy as np
import torch
from torch import nn

from layer_to_layer_pytorch.helpers import zipper
from layer_to_layer_pytorch.types import LossFn


class L2LLoss:
    def __init__(
        self,
        model,
        loss_fn: LossFn,
        # store_grad_on_calc: bool = True,
        **forward_kwargs,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.store_grad_on_calc = False  # store_grad_on_calc
        self.forward_kwargs = forward_kwargs or {}

        self._batch = None
        self._target = None

    @torch.no_grad()
    def __call__(
        self, batch: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        self._batch = batch
        self._target = target

        microbatch_size = self.model._get_microbatch_size(batch)
        num_steps_in_loss = batch.shape[0] // microbatch_size
        losses: List[torch.Tensor] = []

        last_layer: nn.Module = copy.deepcopy(self.model._get_layers()[-1]).to(
            self.model.gpu_device
        )

        for microbatch, microtarget in zipper(
            batch.split(microbatch_size),
            target.split(microbatch_size),
            verbose=False,
            desc="Microbatching",
            total=num_steps_in_loss,
            leave=False,
        ):
            microbatch = microbatch.to(self.model.gpu_device)
            # microbatch.requires_grad = True

            microtarget = microtarget.to(self.model.gpu_device)

            activation: torch.Tensor = last_layer(
                microbatch, **self.forward_kwargs
            )

            loss = self.loss_fn(activation, microtarget)
            losses.append(loss.item())

            # if self.store_grad_on_calc:
            #     loss.backward()
            #     self.model._grads[-1].append(microbatch.grad.cpu())

        with torch.no_grad():
            loss_value = torch.tensor(np.sum(losses) / num_steps_in_loss)

        return loss_value

    def backward(self) -> None:
        self.model.backward(
            self._batch,
            self._target,
            loss_fn=self.loss_fn,
            # skip_last_layer=self.store_grad_on_calc,
        )

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._batch = None
        self._target = None


__all__ = ["L2LLoss"]
