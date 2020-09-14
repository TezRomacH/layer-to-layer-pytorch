# L2L execution algorithm PyTorch

<div align="center">

[![Build status](https://github.com/TezRomacH/layer-to-layer-pytorch/workflows/build/badge.svg?branch=master&event=push)](https://github.com/TezRomacH/layer-to-layer-pytorch/actions?query=workflow%3Abuild)
[![Python Version](https://img.shields.io/pypi/pyversions/layer-to-layer-pytorch.svg)](https://pypi.org/project/layer-to-layer-pytorch/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/TezRomacH/layer-to-layer-pytorch/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/TezRomacH/layer-to-layer-pytorch/blob/master/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%F0%9F%9A%80-semantic%20versions-informational.svg)](https://github.com/TezRomacH/layer-to-layer-pytorch/releases)
[![License](https://img.shields.io/github/license/TezRomacH/layer-to-layer-pytorch)](https://github.com/TezRomacH/layer-to-layer-pytorch/blob/master/LICENSE)

PyTorch implementation of L2L execution algorithm from paper [Training Large Neural Networks with Constant Memory using a New Execution Algorithm](https://arxiv.org/abs/2002.05645)
</div>

## 🚀 Exapmle

You need to define a torch model where all layers are specified in ModuleList.

for example

```python
import torch
from torch import nn, optim

class M(nn.Module):
    def __init__(self, depth: int, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(),
                )
            ]
            + [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(),
                )
                for i in range(depth)
            ]
            + [nn.Linear(hidden_dim, dim), nn.Sigmoid()]
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        x = batch
        for l in self.layers:
            x = l(x)

        return x

```

Then, you can use the L2L wrapper over this model.

```python
from layer_to_layer_pytorch.l2l import Layer2Layer

model = M(depth=5, dim=40).train() # on CPU

l2l_model = Layer2Layer(
    model,
    layers_attr="layers", # attribute with ModuleList
    microbatch_size=100,  # size of a microbatch in a minibatch :) from original paper
    verbose=False  # enable tqdm
)
```

And train it, like torch model (almost):

```python
from tqdm.auto import tqdm, trange

x = torch.rand(1_000, 40) # on CPU
y = torch.rand(1_000, 40) # on CPU

losses = []
loss_fn = nn.MSELoss(reduction="sum") # since L2L calcs average losses itself, we just need to save them

optimizer = optim.AdamW(l2l_model.main_model.parameters(), lr=0.001) # optimizer works with the main model on CPU

for i in trange(5000):
    l2l_model.zero_grad()
    l2l_model.forward(x)

    loss_value = l2l_model.backward(x, y, loss_fn)

    if i % 50 == 0:
        tqdm.write(f"[{i}] loss = {loss_value.item()}")
    losses.append(loss_value.item())

    optimizer.step()
```

## Installation

```bash
pip install layer-to-layer-pytorch
```

or install with `Poetry`

```bash
poetry add layer-to-layer-pytorch
```

## 📈 Releases

You can see the list of available releases on the [GitHub Releases](https://github.com/TezRomacH/layer-to-layer-pytorch/releases) page.

We follow [Semantic Versions](https://semver.org/) specification.

## 🛡 License

[![License](https://img.shields.io/github/license/TezRomacH/layer-to-layer-pytorch)](https://github.com/TezRomacH/layer-to-layer-pytorch/blob/master/LICENSE)

This project is licensed under the terms of the `MIT` license. See [LICENSE](https://github.com/TezRomacH/layer-to-layer-pytorch/blob/master/LICENSE) for more details.

## 📃 Citation

### This library

```
@misc{layer-to-layer-pytorch,
  author = {Roman Tezikov},
  title = {PyTorch implementation of L2L execution algorithm},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/TezRomacH/layer-to-layer-pytorch}}
}
```

### Original paper

```
@article{Pudipeddi2020TrainingLN,
  title={Training Large Neural Networks with Constant Memory using a New Execution Algorithm},
  author={Bharadwaj Pudipeddi and Maral Mesmakhosroshahi and J. Xi and S. Bharadwaj},
  journal={ArXiv},
  year={2020},
  volume={abs/2002.05645}
}
```

## Credits

This project was generated with [`python-package-template`](https://github.com/TezRomacH/python-package-template).
