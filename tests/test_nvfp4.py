"""Small CPU checks for packed E2M1 values and cuBLASLt's block-scale addresses."""

import unittest

import torch

from reference.python.nvfp4 import dequantize, encode_e2m1, pack_scales, quantize, unpack_scales


class NVFP4Tests(unittest.TestCase):
    def test_e2m1_values_and_even_ties(self):
        values = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -.5, -6])
        self.assertEqual(encode_e2m1(values).tolist(), [0, 1, 2, 3, 4, 5, 6, 7, 9, 15])
        midpoints = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5, 100])
        self.assertEqual(encode_e2m1(midpoints).tolist(), [0, 2, 2, 4, 4, 6, 6, 7])

    def test_scale_addresses(self):
        rows, k = 256, 128
        raw = (torch.arange(rows * k // 16) % 126).to(torch.uint8).reshape(rows, k // 16)
        scales = raw.view(torch.float8_e4m3fn)
        packed = pack_scales(scales)
        # Independent cuBLASLt tile address: 128 rows x four K-block scales.
        for row in (0, 1, 31, 32, 127, 128, 255):
            for block in range(k // 16):
                offset = ((row // 128) * (k // 64) + block // 4) * 512
                offset += (row % 32) * 16 + ((row % 128) // 32) * 4 + block % 4
                self.assertEqual(packed[offset].item(), raw[row, block].item())
        torch.testing.assert_close(unpack_scales(packed, rows, k), scales.float())

    def test_packing_round_trip(self):
        values = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6,
                               0, -.5, -1, -1.5, -2, -3, -4, -6]).repeat(128, 4)
        packed, scales = quantize(values)
        self.assertEqual(packed[0, :8].tolist(), [0x10, 0x32, 0x54, 0x76, 0x90, 0xBA, 0xDC, 0xFE])
        torch.testing.assert_close(dequantize(packed, scales), values, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
