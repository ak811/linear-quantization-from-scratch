from __future__ import annotations

import argparse
import os
import torch

from quantization.linear import linear_quantize
from utils.viz import save_linear_quantize_demo_figure


def main():
    parser = argparse.ArgumentParser(description="Demo: linear quantization on a toy tensor.")
    parser.add_argument("--out", type=str, default="assets/linear_quantize_demo.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    test_tensor = torch.tensor([
        [ 0.0523,  0.6364, -0.0968, -0.0020,  0.1940],
        [ 0.7500,  0.5507,  0.6188, -0.1734,  0.4677],
        [-0.0669,  0.3836,  0.4297,  0.6267, -0.0695],
        [ 0.1536, -0.0038,  0.6075,  0.6817,  0.0601],
        [ 0.6446, -0.2500,  0.5376, -0.2226,  0.2333],
    ], dtype=torch.float32)

    bitwidth = 2
    real_min, real_max = -0.25, 0.75
    scale = 1 / 3
    zero_point = -1

    q = linear_quantize(test_tensor, bitwidth=bitwidth, scale=scale, zero_point=zero_point, dtype=torch.int8)
    reconstructed = scale * (q.float() - zero_point)

    expected = torch.tensor([
        [-1,  1, -1, -1,  0],
        [ 1,  1,  1, -2,  0],
        [-1,  0,  0,  1, -1],
        [-1, -1,  1,  1, -1],
        [ 1, -2,  1, -2,  0],
    ], dtype=torch.int8)

    print("* Test linear_quantize()")
    print(f"    target bitwidth: {bitwidth} bits")
    print(f"        scale: {scale}")
    print(f"        zero point: {zero_point}")
    assert q.equal(expected)
    print("* Test passed.")

    save_linear_quantize_demo_figure(
        test_tensor=test_tensor,
        quantized_tensor=q,
        reconstructed_tensor=reconstructed,
        bitwidth=bitwidth,
        real_min=real_min,
        real_max=real_max,
        out_path=args.out,
    )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
