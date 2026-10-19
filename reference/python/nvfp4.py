"""E2M1 packing and cuBLASLt VEC16 UE4M3 scale layout; never timed."""

import torch


def encode_e2m1(values):
    """Round to nearest-even E2M1, saturating at +/-6; return one nibble per value."""
    boundaries = values.new_tensor((.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0))
    magnitude = values.abs().contiguous()
    code = torch.bucketize(magnitude, boundaries)
    # At a midpoint the even code wins, not necessarily the smaller value.
    tie = magnitude == boundaries[code.clamp(max=6)]
    code += tie & ((code & 1) != 0)
    return (code | ((values < 0).to(torch.int64) << 3)).to(torch.uint8)


def pack_scales(scales):
    """[rows,K/16] -> [rows/128,K/64,32,4,4] byte layout required by cuBLASLt."""
    rows, blocks = scales.shape
    return (scales.view(torch.uint8).reshape(rows // 128, 4, 32, blocks // 4, 4)
            .permute(0, 3, 2, 1, 4).contiguous().reshape(-1))


def unpack_scales(scales, rows, k):
    return (scales.reshape(rows // 128, k // 64, 32, 4, 4)
            .permute(0, 3, 2, 1, 4).contiguous().reshape(rows, k // 16)
            .view(torch.float8_e4m3fn).float())


def quantize(values):
    """Quantize each 16-value block using its rounded UE4M3 max/6 scale."""
    rows, k = values.shape
    blocks = values.float().reshape(rows, k // 16, 16)
    scales = (blocks.abs().amax(-1) / 6).clamp(min=2 ** -9, max=448).to(torch.float8_e4m3fn)
    codes = encode_e2m1(blocks / scales.float()[..., None]).reshape(rows, k)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.contiguous(), pack_scales(scales)


def dequantize(packed, scales):
    """Decode the stored bytes, not the original unquantized tensor."""
    rows, half_k = packed.shape
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(rows, 2 * half_k)
    levels = torch.tensor((0, .5, 1, 1.5, 2, 3, 4, 6), device=packed.device)
    values = levels[(codes & 7).long()] * (1 - 2 * (codes >> 3).float())
    scales = unpack_scales(scales, rows, 2 * half_k)
    return (values.reshape(rows, -1, 16) * scales[..., None]).reshape(rows, 2 * half_k)
