from __future__ import annotations

import argparse
import os
import torch

from models.vgg import VGG
from quantization.linear import linear_quantize_weight_per_channel
from utils.viz import save_weight_histogram_grid


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Per-channel linear quantization on VGG weights, then plot histograms.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to VGG state_dict checkpoint (e.g. model_199-1.tar)")
    parser.add_argument("--bitwidth", type=int, default=4, choices=[2, 4, 8], help="Weight quantization bitwidth")
    parser.add_argument("--out", type=str, default="assets/weight_hist_4bit.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    model = VGG()
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state)

    for name, param in model.named_parameters():
        if param.dim() > 1:
            q_w, scale, zp = linear_quantize_weight_per_channel(param.data, args.bitwidth)
            param.copy_(q_w)

    save_weight_histogram_grid(
        model=model,
        title=f"Histogram of Weights (bitwidth={args.bitwidth} bits)",
        out_path=args.out,
        bitwidth=args.bitwidth,
    )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
