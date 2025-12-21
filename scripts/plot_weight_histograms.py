from __future__ import annotations

import argparse
import os
import torch

from models.vgg import VGG
from utils.viz import save_weight_histogram_grid


def main():
    parser = argparse.ArgumentParser(description="Plot VGG weight histograms (grid) for a given bitwidth label.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to VGG state_dict checkpoint (e.g. model_199-1.tar)")
    parser.add_argument("--bitwidth", type=int, default=32, help="Bitwidth label used for binning and title")
    parser.add_argument("--out", type=str, default="assets/weight_hist_fp32.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    model = VGG()
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state)

    save_weight_histogram_grid(
        model=model,
        title=f"Histogram of Weights (bitwidth={args.bitwidth} bits)",
        out_path=args.out,
        bitwidth=args.bitwidth,
    )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
