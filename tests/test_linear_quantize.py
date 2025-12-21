import torch
from quantization.linear import linear_quantize


def test_linear_quantize_reference():
    test_tensor = torch.tensor([
        [ 0.0523,  0.6364, -0.0968, -0.0020,  0.1940],
        [ 0.7500,  0.5507,  0.6188, -0.1734,  0.4677],
        [-0.0669,  0.3836,  0.4297,  0.6267, -0.0695],
        [ 0.1536, -0.0038,  0.6075,  0.6817,  0.0601],
        [ 0.6446, -0.2500,  0.5376, -0.2226,  0.2333],
    ], dtype=torch.float32)

    expected = torch.tensor([
        [-1,  1, -1, -1,  0],
        [ 1,  1,  1, -2,  0],
        [-1,  0,  0,  1, -1],
        [-1, -1,  1,  1, -1],
        [ 1, -2,  1, -2,  0],
    ], dtype=torch.int8)

    q = linear_quantize(test_tensor, bitwidth=2, scale=1/3, zero_point=-1, dtype=torch.int8)
    assert q.equal(expected)
