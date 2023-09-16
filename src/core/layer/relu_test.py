import numpy as np
from .relu import ReluLayer


def test_init():
    layer = ReluLayer(size=3)
    assert layer.size == 3
    assert layer.mask is None


def test_forward():
    layer = ReluLayer(size=3)
    input = np.array([[-1, 0, 1], [-2, 2, 0]])
    output = layer.forward(input)
    assert output.shape == (2, 3)
    assert np.array_equal(output, np.array([[0, 0, 1], [0, 2, 0]]))


def test_backward():
    layer = ReluLayer(size=3)
    input = np.array([[-1, 0, 1], [-2, 2, 0]])
    layer.forward(input)
    backproped_diff = np.array([[1, 0, -1], [-2, 2, 0]])
    output_diff = layer.backward(backproped_diff)
    assert output_diff.shape == (2, 3)
    assert np.array_equal(output_diff, np.array([[0, 0, -1], [0, 2, 0]]))
