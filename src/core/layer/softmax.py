import numpy as np


class SoftmaxWithLossLayer:
    size: int

    output: np.ndarray | None = None
    teacher: np.ndarray | None = None

    def __init__(self, *, size: int) -> None:
        if size < 1:
            raise ValueError("out of valid size range")

        self.size = size

    def forward(self, input: np.ndarray, teacher: np.ndarray) -> np.float64:
        if input.ndim != 2 or teacher.ndim != 2 or input.shape[0] != teacher.shape[0] or input.shape[1] != self.size:
            print(input.shape, teacher.shape, self.size)
            raise ValueError("input or teacher shape is invalid")

        self.output = softmax(input)
        self.teacher = teacher
        error = cross_entropy_error(self.output, self.teacher)
        if np.isnan(error):
            raise ValueError("error is nan")
        return error

    def backward(self) -> np.ndarray:
        if self.output is None or self.teacher is None:
            raise ValueError("must forward before backward")
        if self.output.ndim != 2 or self.teacher.ndim != 2 or self.output.shape[0] != self.teacher.shape[0]:
            print(self.output.shape, self.teacher.shape)
            raise ValueError("output or teacher shape is invalid")

        data_count = self.teacher.shape[0]
        diff = (self.output - self.teacher) / data_count
        return diff


# input: (n, k) -> (n, k)
def softmax(input: np.ndarray) -> np.ndarray:
    if input.ndim != 2:
        raise ValueError("input shape is invalid")
    maxes: np.ndarray = np.max(input, axis=1, keepdims=True)
    exp: np.ndarray = np.exp(input - maxes)  # To prevent overflow
    prob: np.ndarray = exp / np.sum(exp, axis=-1, keepdims=True)
    if input.shape != prob.shape:
        raise ValueError("input.shape != prob.shape")
    return prob


# output: (n, k), teacher: (n, )
def cross_entropy_error(output: np.ndarray, teacher: np.ndarray) -> np.float64:
    if output.ndim != 2 or output.shape != teacher.shape:
        raise ValueError("invalid shape")

    data_count = teacher.shape[0]
    teacher_index = teacher.argmax(axis=1)
    delta = 1e-7  # To prevent np.log from being -inf

    total_error = -np.sum(np.log(output[np.arange(data_count), teacher_index] + delta))
    return total_error / float(data_count)  # average in order to evaluate the loss
