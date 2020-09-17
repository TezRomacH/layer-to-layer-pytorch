# type: ignore[attr-defined]

from typing import Optional

import random
from enum import Enum

import typer

from layer_to_layer_pytorch import __version__

app = typer.Typer(
    name="layer-to-layer-pytorch",
    help="PyTorch implementation of L2L execution algorithm",
    add_completion=False,
)


def version_callback(value: bool):
    """Prints the version of the package."""
    if value:
        print(f"layer-to-layer-pytorch version: {__version__}")
        raise typer.Exit()


@app.command(name="")
def main(
    version: bool = typer.Option(
        None,
        "-v",
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Prints the version of the layer-to-layer-pytorch package.",
    ),
):
    pass
