import torch

from quantization.linear import (
    linear_quantize_feature,
    linear_quantize_weight_per_channel,
    linear_quantize_bias_per_output_channel,
    shift_quantized_linear_bias,
    quantized_linear,
)


def test_quantized_fc_reference():
    input_fp = torch.tensor([
        [0.6118, 0.7288, 0.8511, 0.2849, 0.8427, 0.7435, 0.4014, 0.2794],
        [0.3676, 0.2426, 0.1612, 0.7684, 0.6038, 0.0400, 0.2240, 0.4237],
        [0.6565, 0.6878, 0.4670, 0.3470, 0.2281, 0.8074, 0.0178, 0.3999],
        [0.1863, 0.3567, 0.6104, 0.0497, 0.0577, 0.2990, 0.6687, 0.8626],
    ], dtype=torch.float32)

    weight_fp = torch.tensor([
        [ 1.2626e-01, -1.4752e-01,  8.1910e-02,  2.4982e-01, -1.0495e-01,
         -1.9227e-01, -1.8550e-01, -1.5700e-01],
        [ 2.7624e-01, -4.3835e-01,  5.1010e-02, -1.2020e-01, -2.0344e-01,
          1.0202e-01, -2.0799e-01,  2.4112e-01],
        [-3.8216e-01, -2.8047e-01,  8.5238e-02, -4.2504e-01, -2.0952e-01,
          3.2018e-01, -3.3619e-01,  2.0219e-01],
        [ 8.9233e-02, -1.0124e-01,  1.1467e-01,  2.0091e-01,  1.1438e-01,
         -4.2427e-01,  1.0178e-01, -3.0941e-04],
        [-1.8837e-02, -2.1256e-01, -4.5285e-01,  2.0949e-01, -3.8684e-01,
         -1.7100e-01, -4.5331e-01, -2.0433e-01],
        [-2.0038e-01, -5.3757e-02,  1.8997e-01, -3.6866e-01,  5.5484e-02,
          1.5643e-01, -2.3538e-01,  2.1103e-01],
        [-2.6875e-01,  2.4984e-01, -2.3514e-01,  2.5527e-01,  2.0322e-01,
          3.7675e-01,  6.1563e-02,  1.7201e-01],
        [ 3.3541e-01, -3.3555e-01, -4.3349e-01,  4.3043e-01, -2.0498e-01,
         -1.8366e-01, -9.1553e-02, -4.1168e-01],
    ], dtype=torch.float32)

    bias_fp = torch.tensor([0.1954, -0.2756, 0.3113, 0.1149, 0.4274, 0.2429, -0.1721, -0.2502], dtype=torch.float32)
    output_fp = torch.nn.functional.linear(input_fp, weight_fp, bias_fp)

    quantized_bias_ref = torch.tensor([ 3, -2,  3,  1,  3,  2, -2, -2], dtype=torch.int32)
    shifted_bias_ref = torch.tensor([-1,  0, -3, -1, -3,  0,  2, -4], dtype=torch.int32)
    out_ref = torch.tensor([
        [ 0, -1,  0, -1, -1,  0,  1, -2],
        [ 0,  0, -1,  0,  0,  0,  0, -1],
        [ 0,  0,  0, -1,  0,  0,  0, -1],
        [ 0,  0,  0,  0,  0,  1, -1, -2],
    ], dtype=torch.int8)

    bitwidth = 2

    q_weight, weight_scale, weight_zp = linear_quantize_weight_per_channel(weight_fp, bitwidth)
    q_input, input_scale, input_zp = linear_quantize_feature(input_fp, bitwidth)
    q_bias, bias_scale, bias_zp = linear_quantize_bias_per_output_channel(bias_fp, weight_scale, input_scale)

    assert q_bias.equal(quantized_bias_ref)
    shifted = shift_quantized_linear_bias(q_bias, q_weight, input_zp)
    assert shifted.equal(shifted_bias_ref)

    q_output, output_scale, output_zp = linear_quantize_feature(output_fp, bitwidth)

    calc_q_out = quantized_linear(
        input_q=q_input,
        weight_q=q_weight,
        bias_q_shifted=shifted,
        feature_bitwidth=bitwidth,
        weight_bitwidth=bitwidth,
        input_zero_point=input_zp,
        output_zero_point=output_zp,
        input_scale=input_scale,
        weight_scale=weight_scale.view(-1),
        output_scale=output_scale,
    )
    assert calc_q_out.equal(out_ref)
