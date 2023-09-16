import pytest
import numpy as np
from .affine import AffineLayer


def test_init() -> None:
    layer = AffineLayer(input_size=3, output_size=2)
    assert layer.weight.shape == (3, 2)
    assert layer.bias.shape == (2,)
    assert layer.weight_diff is None
    assert layer.bias_diff is None


def test_forward() -> None:
    # Set weight and bias manually
    layer = AffineLayer(input_size=3, output_size=2)
    layer.weight = np.array([[1, 2], [3, 4], [5, 6]])
    layer.bias = np.array([7, 8])

    # forward
    input = np.array([[1, 2, 3], [4, 5, 6]])
    actual = layer.forward(input)

    # assert
    assert layer.input is not None and np.array_equal(layer.input, input)  # input must be set
    assert actual.shape == (2, 2)
    want = np.array(
        [
            [1 * 1 + 2 * 3 + 3 * 5 + 7, 1 * 2 + 2 * 4 + 3 * 6 + 8],
            [4 * 1 + 5 * 3 + 6 * 5 + 7, 4 * 2 + 5 * 4 + 6 * 6 + 8],
        ]
    )
    assert np.array_equal(want, actual)


def test_backward() -> None:
    # Set weight and bias manually
    layer = AffineLayer(input_size=3, output_size=2)
    layer.weight = np.array([[1, 2], [3, 4], [5, 6]])
    layer.bias = np.array([7, 8])

    # forward (backword will be called after forwarding)
    input = np.array([[1, 2, 3], [4, 5, 6]])
    layer.forward(input)

    # backward
    backproped_diff = np.array([[1, 2], [3, 4]])
    output_diff = layer.backward(backproped_diff)

    # assert
    assert output_diff.shape == (2, 3)

    if (weight_diff := layer.weight_diff) is not None:
        assert weight_diff.shape == (3, 2)
    else:
        pytest.fail("layer.weight_diff is not set")

    if (bias_diff := layer.bias_diff) is not None:
        assert bias_diff.shape == (2,)
    else:
        pytest.fail("layer.bias_diff is not set")
