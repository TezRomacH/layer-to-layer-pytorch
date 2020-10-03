from typing import Callable, Union

import torch

Device = Union[str, torch.device]
Criterion = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

__all__ = ["Device", "Criterion"]
