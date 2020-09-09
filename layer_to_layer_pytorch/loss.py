# from typing import Callable

# import torch
# from torch import nn

# from layer_to_layer_pytorch.types import LossFn
# from layer_to_layer_pytorch.l2l import Layer2Layer

# class L2LLoss:
#     def __init__(self, model: Layer2Layer, loss_fn: LossFn):
#         self.model = model
#         self.loss_fn = loss_fn

#     def __call__(self, batch: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
