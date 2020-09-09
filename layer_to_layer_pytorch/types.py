from typing import Callable, List, Union

import torch

Device = Union[str, torch.device]

TensorOrTensorArray = Union[torch.Tensor, List[torch.Tensor]]

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

__all__ = ["Device", "TensorOrTensorArray", "LossFn"]
