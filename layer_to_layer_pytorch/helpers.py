from typing import Iterable

from tqdm.auto import tqdm
from tqdm.contrib import tenumerate, tzip


def iterator(iterable: Iterable, verbose: bool, **kwargs):
    if not verbose:
        return iterable

    return tqdm(iterable, **kwargs)


def enumerator(iterable: Iterable, verbose: bool, **kwargs):
    if not verbose:
        return enumerate(iterable)

    return tenumerate(iterable, **kwargs)


def zipper(iterable1: Iterable, iterable2, verbose: bool, **kwargs):
    if not verbose:
        return zip(iterable1, iterable2)

    return tzip(iterable1, iterable2, **kwargs)


__all__ = ["iterator", "enumerator", "zipper"]
