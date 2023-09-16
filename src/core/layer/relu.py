import numpy as np
from typing import Any


class ReluLayer:
    size: int
    mask: np.ndarray[Any, np.dtype[np.bool_]] | None = None

    def __init__(self, *, size: int) -> None:
        if size < 1:
            raise ValueError("out of valid size range")

        self.size = size

    def forward(self, input: np.ndarray) -> np.ndarray:
        data_count, input_size = input.shape
        if input_size != self.size:
            raise ValueError("invalid input")

        self.mask = input <= 0
        output = input.copy()
        output[self.mask] = 0
        return output

    def backward(self, backwarded_diff: np.ndarray) -> np.ndarray:
        if self.mask is None:
            raise ValueError("forward before backward")
        if self.mask.shape[0] != backwarded_diff.shape[0]:
            raise ValueError("invalid shape")

        diff = backwarded_diff.copy()
        diff[self.mask] = 0
        return diff
