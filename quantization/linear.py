from __future__ import annotations

from typing import Tuple, Union
import torch


def get_quantized_range(bitwidth: int) -> Tuple[int, int]:
    """Signed two's-complement integer range for bitwidth."""
    quantized_max = (1 << (bitwidth - 1)) - 1
    quantized_min = -(1 << (bitwidth - 1))
    return quantized_min, quantized_max


def linear_quantize(
    fp_tensor: torch.Tensor,
    bitwidth: int,
    scale: Union[float, torch.Tensor],
    zero_point: Union[int, torch.Tensor],
    dtype: torch.dtype = torch.int8,
) -> torch.Tensor:
    """
    Linear quantization:
        fp = scale * (q - zero_point)
      => q = round(fp / scale) + zero_point

    Returns:
        integer tensor (dtype) clamped into bitwidth range.
    """
    assert fp_tensor.dtype == torch.float32 or fp_tensor.dtype == torch.float
    if isinstance(scale, torch.Tensor):
        assert scale.dtype == torch.float32 or scale.dtype == torch.float
    if isinstance(zero_point, torch.Tensor):
        assert zero_point.dtype == dtype

    # Step 1: scale
    scaled = fp_tensor / scale

    # Step 2: round and cast
    rounded = torch.round(scaled).to(dtype)

    # Step 3: add zero point
    shifted = rounded + zero_point

    # Step 4: clamp
    qmin, qmax = get_quantized_range(bitwidth)
    return shifted.clamp(qmin, qmax)


def get_quantization_scale_and_zero_point(fp_tensor: torch.Tensor, bitwidth: int) -> Tuple[float, int]:
    """
    Derive scale and zero point mapping fp range -> quantized integer range.
      scale = (fp_max - fp_min) / (qmax - qmin)
      zero_point = round(qmin - fp_min / scale)
    """
    qmin, qmax = get_quantized_range(bitwidth)
    fp_max = fp_tensor.max().item()
    fp_min = fp_tensor.min().item()

    scale = (fp_max - fp_min) / float(qmax - qmin) if fp_max != fp_min else 1.0
    zero_point = qmin - fp_min / scale

    if zero_point < qmin:
        zero_point = qmin
    elif zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = round(zero_point)

    return float(scale), int(zero_point)


def linear_quantize_feature(fp_tensor: torch.Tensor, bitwidth: int):
    """Quantize an activation/feature tensor with derived (scale, zero_point)."""
    scale, zero_point = get_quantization_scale_and_zero_point(fp_tensor, bitwidth)
    q = linear_quantize(fp_tensor, bitwidth, scale, zero_point, dtype=torch.int8)
    return q, scale, zero_point


def get_quantization_scale_for_weight(weight: torch.Tensor, bitwidth: int) -> float:
    """
    Weight quantization commonly uses symmetric quantization (zero_point=0).
    scale = max(|w|) / qmax
    """
    fp_max = max(weight.abs().max().item(), 5e-7)
    _, qmax = get_quantized_range(bitwidth)
    return float(fp_max / qmax)


def linear_quantize_weight_per_channel(weight: torch.Tensor, bitwidth: int):
    """
    Per-output-channel weight quantization for Conv/Linear weights.
    Assumes output-channel dimension is dim 0.

    Returns:
      q_weight (int8), scale (float tensor broadcastable), zero_point=0
    """
    dim_oc = 0
    num_oc = weight.shape[dim_oc]
    scale = torch.zeros(num_oc, device=weight.device, dtype=torch.float32)

    for oc in range(num_oc):
        subt = weight.select(dim_oc, oc)
        scale[oc] = get_quantization_scale_for_weight(subt, bitwidth)

    shape = [1] * weight.dim()
    shape[dim_oc] = -1
    scale = scale.view(shape)  # broadcastable
    q_weight = linear_quantize(weight, bitwidth, scale, zero_point=0, dtype=torch.int8)
    return q_weight, scale, 0


def linear_quantize_bias_per_output_channel(
    bias: torch.Tensor,
    weight_scale: Union[float, torch.Tensor],
    input_scale: float,
):
    """
    Bias quantization:
      bias_scale = input_scale * weight_scale
      q_bias = round(bias / bias_scale)  (int32)
    """
    assert bias.dim() == 1
    assert bias.dtype == torch.float32 or bias.dtype == torch.float
    assert isinstance(input_scale, float)

    if isinstance(weight_scale, torch.Tensor):
        ws = weight_scale.view(-1)
        assert ws.dtype == torch.float32 or ws.dtype == torch.float
        assert ws.numel() == bias.numel()
        bias_scale = input_scale * ws
    else:
        bias_scale = input_scale * float(weight_scale)

    q_bias = linear_quantize(bias, 32, bias_scale, zero_point=0, dtype=torch.int32)
    return q_bias, bias_scale, 0


def shift_quantized_linear_bias(quantized_bias: torch.Tensor, quantized_weight: torch.Tensor, input_zero_point: int):
    """
    Incorporate input zero point:
      Q_bias = q_bias - Linear(Z_input, q_weight)
    For Linear: Linear(Z_input, q_weight) = Z_input * sum(q_weight over in_features).
    """
    assert quantized_bias.dtype == torch.int32
    assert quantized_weight.dtype == torch.int8
    assert isinstance(input_zero_point, int)

    return quantized_bias - quantized_weight.sum(1).to(torch.int32) * input_zero_point


def quantized_linear(
    input_q: torch.Tensor,
    weight_q: torch.Tensor,
    bias_q_shifted: torch.Tensor,
    feature_bitwidth: int,
    weight_bitwidth: int,
    input_zero_point: int,
    output_zero_point: int,
    input_scale: float,
    weight_scale: torch.Tensor,
    output_scale: float,
) -> torch.Tensor:
    """
    Quantized fully-connected layer inference:

      q_out = (Linear[q_in, q_w] + Q_bias) * (S_in * S_w / S_out) + Z_out

    - int8 mult with int32 accumulation on CPU
    - output is clamped to feature bitwidth and returned as int8
    """
    assert input_q.dtype == torch.int8
    assert weight_q.dtype == torch.int8
    assert bias_q_shifted is None or bias_q_shifted.dtype == torch.int32
    assert isinstance(input_zero_point, int)
    assert isinstance(output_zero_point, int)
    assert isinstance(input_scale, float)
    assert isinstance(output_scale, float)
    assert weight_scale.dtype == torch.float32 or weight_scale.dtype == torch.float

    if input_q.device.type == "cpu":
        out_int = torch.nn.functional.linear(input_q.to(torch.int32), weight_q.to(torch.int32), bias_q_shifted)
    else:
        out_int = torch.nn.functional.linear(input_q.float(), weight_q.float(), bias_q_shifted.float()).to(torch.float32)

    combined = (input_scale * weight_scale / output_scale).view(1, -1)
    out_fp = out_int.to(torch.float32) * combined
    out_fp = out_fp + float(output_zero_point)

    qmin, qmax = get_quantized_range(feature_bitwidth)
    return out_fp.round().clamp(qmin, qmax).to(torch.int8)
