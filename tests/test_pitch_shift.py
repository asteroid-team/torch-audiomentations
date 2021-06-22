import unittest

import numpy as np
import pytest
import torch
from numpy.testing import assert_almost_equal
from torch_audiomentations import PitchShift


def get_example(device_name):
    return torch.rand(size=(8, 2, 32000), dtype=torch.float32, device=device_name) - 0.5


class TestPitchShift(unittest.TestCase):
    @pytest.mark.parametrize(
        "device_name",
        [
            pytest.param("cpu"),
            pytest.param(
                "cuda",
                marks=pytest.mark.skip("Requires CUDA")
                if not torch.cuda.is_available()
                else [],
            ),
        ],
    )
    def test_per_batch_shift(self, device_name):
        samples = get_example(device_name)
        aug = PitchShift(16000, p=1, mode="per_example")
        aug.randomize_parameters(samples)
        results = aug.apply_transform(samples)
        self.assertEqual(results.shape, samples.shape)

    def test_per_example_shift(self, device_name):
        samples = get_example(device_name)
        aug = PitchShift(16000, p=1, mode="per_channel")
        aug.randomize_parameters(samples)
        results = aug.apply_transform(samples)
        self.assertEqual(results.shape, samples.shape)

    def test_per_channel_shift(self, device_name):
        samples = get_example(device_name)
        aug = PitchShift(16000, p=1, mode="per_batch")
        aug.randomize_parameters(samples)
        results = aug.apply_transform(samples)
        self.assertEqual(results.shape, samples.shape)
