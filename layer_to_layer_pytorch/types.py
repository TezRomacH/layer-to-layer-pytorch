from typing import List, Union

import torch

Device = Union[str, torch.device]

TensorOrTensorArray = Union[torch.Tensor, List[torch.Tensor]]

__all__ = ["Device", "TensorOrTensorArray"]
