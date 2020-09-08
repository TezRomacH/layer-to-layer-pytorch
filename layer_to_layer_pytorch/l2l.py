from typing import Callable, List

import copy

import torch
from torch import nn

from layer_to_layer_pytorch.types import Device, TensorOrTensorArray

# from tqdm.auto import tqdm
# from tqdm.contrib import tenumerate, tzip


class Layer2Layer:
    def __init__(
        self,
        model: nn.Module,
        layers_field: str,
        microbatch_size: int,
        gpu_device: Device = "cuda",
        verbose: bool = False,
    ):
        layers = getattr(model, layers_field, None)
        if (layers is None) or (not isinstance(layers, nn.ModuleList)):
            raise ValueError(
                f"Model must contain `nn.ModuleList` in field `{layers_field}`"
            )

        if microbatch_size < 0:
            raise ValueError(
                f"Size of a microbatch must be greater than zero. Got {microbatch_size}"
            )

        self.main_model: nn.Module = copy.deepcopy(model.cpu())
        self.layers_field: str = layers_field
        self.microbatch_size: int = microbatch_size
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
        layers: nn.ModuleList = getattr(self.main_model, self.layers_field)

        # layer by layer forward pass. only activations are stored
        for idx, l in enumerate(layers):
            layer = copy.deepcopy(l).to(self.gpu_device)

            input: torch.Tensor
            if idx == 0:
                input = batch
            else:
                input = self._activations[idx - 1]

            # forward with microbatching
            for microbatch in input.split(self.microbatch_size):
                microbatch = microbatch.to(self.gpu_device)
                activation: torch.Tensor = layer(microbatch, **kwargs)

                self._activations[idx].append(activation.cpu())

            self._activations[idx] = torch.cat(self._activations[idx], dim=0)

        return self._activations[-1]

    def calculate_gradients(
        self,
        batch: torch.Tensor,
        target: torch.Tensor,
        loss_fn: Callable,
        loss_kwargs: dict = None,
        **forward_kwargs,
    ):
        if loss_kwargs is None:
            loss_kwargs = {}
        # layer by layer backward pass (in reverse order)
        layers: nn.ModuleList = getattr(self.main_model, self.layers_field)

        for idx, l in enumerate(reversed(layers)):
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

            num_steps = len(input) // self.microbatch_size

            # backward with microbatching
            for microbatch, microtarget in zip(
                input.split(self.microbatch_size),
                output.split(self.microbatch_size),
            ):
                microbatch = microbatch.to(self.gpu_device)
                microbatch.requires_grad = True
                activation: torch.Tensor = layer(microbatch, **forward_kwargs)

                if idx == 0:
                    loss = loss_fn(activation, microtarget, **loss_kwargs)
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

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        return self.forward(batch)


__all__ = ["Layer2Layer"]
