import unittest
import torch
import pytest
import numpy as np
from torch_audiomentations.augmentations.random_crop import RandomCrop


class TestRandomCrop(unittest.TestCase):
    def test_crop(self):
        samples = torch.rand(size=(8, 2, 32000), dtype=torch.float32)
        sampling_rate = 16000
        crop_to = 1.5
        desired_samples_len = sampling_rate * crop_to
        Crop = RandomCrop(max_length=crop_to, sampling_rate=sampling_rate)
        cropped_samples = Crop(samples)

        self.assertEqual(desired_samples_len, cropped_samples.size(-1))

    def test_crop_larger_cropto(self):
        samples = torch.rand(size=(8, 2, 32000), dtype=torch.float32)
        sampling_rate = 16000
        crop_to = 3
        Crop = RandomCrop(max_length=crop_to, sampling_rate=sampling_rate)
        cropped_samples = Crop(samples)

        np.testing.assert_array_equal(samples, cropped_samples)
        self.assertEqual(samples.size(-1), cropped_samples.size(-1))

    @pytest.mark.skip(reason="output_type is not implemented yet")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_crop_on_device_cuda(self):
        samples = torch.rand(
            size=(8, 2, 32000), dtype=torch.float32, device=torch.device("cuda")
        )
        sampling_rate = 16000
        crop_to = 1.5
        desired_samples_len = sampling_rate * crop_to
        Crop = RandomCrop(
            max_length=crop_to, sampling_rate=sampling_rate, output_type="dict"
        )
        cropped_samples = Crop(samples)

        self.assertEqual(desired_samples_len, cropped_samples.size(-1))
