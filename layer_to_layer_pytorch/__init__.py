# type: ignore[attr-defined]
"""PyTorch implementation of L2L execution algorithm"""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version


try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

from layer_to_layer_pytorch.l2l import Layer2Layer
from layer_to_layer_pytorch.types import Device, TensorOrTensorArray
