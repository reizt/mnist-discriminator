import numpy as np


class AffineLayer:
    input_size: int
    output_size: int
    weight: np.ndarray  # input_size * output_size
    bias: np.ndarray  # 1 * output_size

    input: np.ndarray | None = None  # data_count * input_size
    weight_diff: np.ndarray | None = None  # input_size * output_size
    bias_diff: np.ndarray | None = None  # 1 * output_size

    def __init__(self, *, input_size: int, output_size: int) -> None:
        if input_size < 1 or output_size < 1:
            raise ValueError("out of valid range")

        self.input_size = input_size
        self.output_size = output_size
        # self.weight = he_initial_weight(width=self.input_size, height=self.output_size)
        self.weight = 0.01 * np.random.randn(self.input_size, self.output_size)
        self.bias = np.zeros(self.output_size)

    def forward(self, input: np.ndarray) -> np.ndarray:
        data_count, input_size = input.shape
        if input_size != self.input_size:
            raise ValueError("invalid shape")

        self.input = input.reshape(data_count, -1)
        return np.dot(self.input, self.weight) + self.bias

    def backward(self, backwarded_diff: np.ndarray) -> np.ndarray:
        if self.input is None:
            raise ValueError("input is none")
        if backwarded_diff.ndim != 2 or backwarded_diff.shape[1] != self.output_size:
            print(backwarded_diff.shape)
            raise ValueError("invalid shape")

        self.weight_diff: np.ndarray = np.dot(self.input.T, backwarded_diff)
        self.bias_diff: np.ndarray = np.sum(backwarded_diff, axis=0)
        return np.dot(backwarded_diff, self.weight.T)

    def update_params(self, learning_rate: float) -> None:
        if self.weight_diff is None or self.bias_diff is None:
            raise ValueError("forward before update params")

        self.weight -= self.weight_diff * learning_rate
        self.bias -= self.bias_diff * learning_rate


def he_initial_weight(width: int, height: int) -> np.ndarray:
    mean = 0
    div = np.sqrt(2 / width)
    return np.random.normal(mean, div, size=(width, height))
