from __future__ import annotations

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from quantization.linear import get_quantized_range


def plot_matrix(ax, tensor: torch.Tensor, title: str, vmin=None, vmax=None, cmap=ListedColormap(["white"])):
    ax.imshow(tensor.cpu().numpy(), vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            datum = tensor[i, j].item()
            if isinstance(datum, float):
                ax.text(j, i, f"{datum:.2f}", ha="center", va="center", color="k", fontsize=8)
            else:
                ax.text(j, i, f"{datum}", ha="center", va="center", color="k", fontsize=8)


def save_linear_quantize_demo_figure(
    test_tensor: torch.Tensor,
    quantized_tensor: torch.Tensor,
    reconstructed_tensor: torch.Tensor,
    bitwidth: int,
    real_min: float,
    real_max: float,
    out_path: str,
):
    qmin, qmax = get_quantized_range(bitwidth)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    plot_matrix(axes[0], test_tensor, "original tensor", vmin=real_min, vmax=real_max)
    plot_matrix(axes[1], quantized_tensor, f"{bitwidth}-bit quantized tensor", vmin=qmin, vmax=qmax, cmap="tab20c")
    plot_matrix(axes[2], reconstructed_tensor, "reconstructed tensor", vmin=real_min, vmax=real_max, cmap="tab20c")
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def save_weight_histogram_grid(model, title: str, out_path: str, bitwidth: int = 32):
    if bitwidth <= 8:
        qmin, qmax = get_quantized_range(bitwidth)
        bins = np.arange(qmin, qmax + 2)
        align = "left"
    else:
        bins = 256
        align = "mid"

    fig, axes = plt.subplots(3, 3, figsize=(10, 6))
    axes = axes.ravel()
    plot_index = 0

    for name, param in model.named_parameters():
        if param.dim() > 1 and plot_index < 9:
            ax = axes[plot_index]
            ax.hist(
                param.detach().view(-1).cpu().numpy(),
                bins=bins,
                density=True,
                align=align,
                alpha=0.5,
                edgecolor="black" if bitwidth <= 4 else None,
            )
            if bitwidth <= 4:
                qmin, qmax = get_quantized_range(bitwidth)
                ax.set_xticks(np.arange(start=qmin, stop=qmax + 1))
            ax.set_xlabel(name, fontsize=8)
            ax.set_ylabel("density", fontsize=8)
            plot_index += 1

    for j in range(plot_index, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_quantized_fc_demo_figure(
    weight: torch.Tensor,
    input_: torch.Tensor,
    output: torch.Tensor,
    q_weight: torch.Tensor,
    q_input: torch.Tensor,
    q_output_calc: torch.Tensor,
    r_weight: torch.Tensor,
    r_input: torch.Tensor,
    r_output: torch.Tensor,
    bitwidth: int,
    out_path: str,
):
    qmin, qmax = get_quantized_range(bitwidth)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    plot_matrix(axes[0, 0], weight, "original weight", vmin=-0.5, vmax=0.5)
    plot_matrix(axes[1, 0], input_.t(), "original input", vmin=0.0, vmax=1.0)
    plot_matrix(axes[2, 0], output.t(), "original output", vmin=-1.5, vmax=1.5)

    plot_matrix(axes[0, 1], q_weight, f"{bitwidth}-bit quantized weight", vmin=qmin, vmax=qmax, cmap="tab20c")
    plot_matrix(axes[1, 1], q_input.t(), f"{bitwidth}-bit quantized input", vmin=qmin, vmax=qmax, cmap="tab20c")
    plot_matrix(axes[2, 1], q_output_calc.t(), "quantized output (quantized_linear)", vmin=qmin, vmax=qmax, cmap="tab20c")

    plot_matrix(axes[0, 2], r_weight, "reconstructed weight", vmin=-0.5, vmax=0.5, cmap="tab20c")
    plot_matrix(axes[1, 2], r_input.t(), "reconstructed input", vmin=0.0, vmax=1.0, cmap="tab20c")
    plot_matrix(axes[2, 2], r_output.t(), "reconstructed output", vmin=-1.5, vmax=1.5, cmap="tab20c")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
