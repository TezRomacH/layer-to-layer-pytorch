from typing import List, Optional

import copy

import numpy as np
import torch
from torch import nn

from layer_to_layer_pytorch.helpers import enumerator, iterator, zipper
from layer_to_layer_pytorch.types import Device, LossFn, TensorOrTensorArray


class Layer2Layer:
    def __init__(
        self,
        model: nn.Module,
        layers_attr: str,
        microbatch_size: Optional[int],
        gpu_device: Device = "cuda",
        verbose: bool = False,
    ):
        layers = getattr(model, layers_attr, None)
        if (layers is None) or (not isinstance(layers, nn.ModuleList)):
            raise ValueError(
                f"Model must contain `nn.ModuleList` in attribute `{layers_attr}`"
            )

        if (microbatch_size is not None) and (microbatch_size < 0):
            raise ValueError(
                f"Size of a microbatch must be greater than zero. Got {microbatch_size}"
            )

        self.main_model: nn.Module = model.cpu()
        self.layers_attr: str = layers_attr
        self.microbatch_size: Optional[int] = microbatch_size
        self.gpu_device: Device = gpu_device

        self.verbose: bool = verbose

        self.num_layers: int = len(layers)

        self._activations: List[TensorOrTensorArray] = [[]] * self.num_layers
        self._grads: List[TensorOrTensorArray] = [[]] * self.num_layers

    def _reset_activations(self):
        self._activations = [[]] * self.num_layers
        self._grads = [[]] * self.num_layers

    def zero_grad(self) -> None:
        for param in self.main_model.parameters():
            param.grad = None

        self._reset_activations()

    @torch.no_grad()
    def forward(self, batch: torch.Tensor, **kwargs) -> torch.Tensor:
        layers: nn.ModuleList = getattr(self.main_model, self.layers_attr)

        # layer by layer forward pass. only activations are stored
        for idx, l in enumerator(
            layers,
            verbose=self.verbose,
            desc="Layers",
            total=self.num_layers,
            leave=False,
        ):
            layer = copy.deepcopy(l).to(self.gpu_device)

            input: torch.Tensor
            if idx == 0:
                input = batch
            else:
                input = self._activations[idx - 1]

            # forward with microbatching
            batch_size = input.shape[0]
            microbatch_size = (
                batch_size
                if self.microbatch_size is None
                else self.microbatch_size
            )
            num_steps: int = input.shape[0] // microbatch_size

            for microbatch in iterator(
                input.split(microbatch_size),
                verbose=False,  # self.verbose,
                desc="Microbatching",
                total=num_steps,
                leave=False,
            ):
                microbatch = microbatch.to(self.gpu_device)
                activation: torch.Tensor = layer(microbatch, **kwargs)

                self._activations[idx].append(activation.cpu())

            self._activations[idx] = torch.cat(self._activations[idx], dim=0)

        return self._activations[-1]

    def calculate_gradients(
        self,
        batch: torch.Tensor,
        target: torch.Tensor,
        loss_fn: LossFn,
        loss_kwargs: dict = None,
        **forward_kwargs,
    ) -> torch.Tensor:
        if loss_kwargs is None:
            loss_kwargs = {}
        # layer by layer backward pass (in reverse order)
        layers: nn.ModuleList = getattr(self.main_model, self.layers_attr)
        losses: List[torch.Tensor] = []
        num_steps_in_loss: int = 1

        for idx, l in enumerator(
            reversed(layers),
            verbose=self.verbose,
            desc="Reverse Layers",
            total=self.num_layers,
            leave=False,
        ):
            layer = copy.deepcopy(l).to(self.gpu_device)
            f_idx: int = self.num_layers - idx - 1

            for param in layer.parameters():
                param.requires_grad = True

            input: torch.Tensor
            output: torch.Tensor

            if f_idx == 0:
                input = batch
                output = self._grads[idx - 1]
            else:
                input = self._activations[f_idx - 1]
                output = target

            batch_size = input.shape[0]
            microbatch_size = (
                batch_size
                if self.microbatch_size is None
                else self.microbatch_size
            )

            num_steps: int = input.shape[0] // microbatch_size
            if idx == 0:
                num_steps_in_loss = num_steps

            # backward with microbatching
            for microbatch, microtarget in zipper(
                input.split(microbatch_size),
                output.split(microbatch_size),
                verbose=False,  # self.verbose,
                desc="Microbatching",
                total=num_steps,
                leave=False,
            ):
                microbatch = microbatch.to(self.gpu_device)
                microbatch.requires_grad = True

                microtarget = microtarget.to(self.gpu_device)

                activation: torch.Tensor = layer(microbatch, **forward_kwargs)

                if idx == 0:
                    loss = loss_fn(activation, microtarget, **loss_kwargs)
                    losses.append(loss.item())
                    loss.backward()

                    self._grads[idx].append(microbatch.grad.cpu())
                else:
                    activation.backward(microtarget)
                    self._grads[idx].append(microbatch.grad.cpu())

            for local_param, main_param in zip(
                layer.parameters(), layers[f_idx].parameters()
            ):
                if main_param.grad is None:
                    main_param.grad = local_param.grad.cpu() / num_steps
                else:
                    main_param.grad += local_param.grad.cpu() / num_steps

            with torch.no_grad():
                self._grads[idx] = (
                    torch.cat(self._grads[idx], dim=0).cpu() / num_steps
                )

        self._grads = list(reversed(self._grads))
        with torch.no_grad():
            loss_value = torch.tensor(np.sum(losses) / num_steps_in_loss)

        return loss_value

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        return self.forward(batch)


__all__ = ["Layer2Layer"]
