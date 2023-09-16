import numpy as np
from .layer.affine import AffineLayer
from .layer.relu import ReluLayer
from .layer.softmax import SoftmaxWithLossLayer


class Network:
    # l: layer, a: activator, al: activator-last
    l1: AffineLayer
    a1: ReluLayer
    l2: AffineLayer
    al: SoftmaxWithLossLayer

    def __init__(self, *, input_size: int) -> None:
        if input_size < 1:
            raise ValueError("out of valid size range")

        self.l1 = AffineLayer(input_size=input_size, output_size=50)  # input_size -> 50
        self.a1 = ReluLayer(size=self.l1.output_size)
        self.l2 = AffineLayer(input_size=self.l1.output_size, output_size=10)  # 50 -> 10
        self.al = SoftmaxWithLossLayer(size=self.l2.output_size)

    def descriminate(self, input: np.ndarray) -> np.ndarray:
        output = self.l1.forward(input)
        output = self.a1.forward(output)
        output = self.l2.forward(output)
        return output

    def forward(self, input: np.ndarray, teacher: np.ndarray) -> np.float64:
        output = self.descriminate(input)
        return self.al.forward(output, teacher)

    def backward(self) -> np.ndarray:
        diff = self.al.backward()
        diff = self.l2.backward(diff)
        diff = self.a1.backward(diff)
        diff = self.l1.backward(diff)
        return diff

    def train(self, input: np.ndarray, teacher: np.ndarray, *, leaning_rate: float) -> np.float64:
        if input.ndim != 2:
            raise ValueError("input ndim must be 2")
        if teacher.ndim != 1:
            raise ValueError("teacher ndim must be 1")
        if input.shape[0] != teacher.shape[0]:
            raise ValueError("input and teacher must have the same size")
        if leaning_rate <= 0:
            raise ValueError("learning_rate must be larger than 0")

        one_hot_teacher = np.eye(10)[teacher]

        loss = self.forward(input, one_hot_teacher)
        self.backward()
        self.l1.update_params(leaning_rate)
        self.l2.update_params(leaning_rate)
        return loss

    def test_accuracy(self, input: np.ndarray, answer: np.ndarray) -> float:
        likelihoods = self.descriminate(input)
        prediction: np.ndarray = np.argmax(likelihoods, axis=1)
        data_count = likelihoods.shape[0]
        correct_count: int = np.sum(prediction == answer)
        return correct_count / float(data_count)
