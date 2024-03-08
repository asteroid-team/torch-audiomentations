import unittest

import numpy as np
import pytest
import torch
from numpy.testing import assert_almost_equal
from torch_audiomentations import Delay


def get_example():
    return (
        torch.rand(
            size=(8, 2, 32000),
            dtype=torch.float32,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        - 0.5
    )


class TestDelay(unittest.TestCase):
    def test_per_example_delay(self):
        samples = get_example()
        aug = Delay(sample_rate=16000, p=1, mode="per_example", output_type="dict")
        aug.randomize_parameters(samples)
        results = aug.apply_transform(samples).samples
        self.assertEqual(results.shape, samples.shape)
        self.assertEqual(results.dtype, torch.float32)

    def test_per_batch_shift(self):
        samples = get_example()
        aug = Delay(sample_rate=16000, p=1, mode="per_batch", output_type="dict")
        aug.randomize_parameters(samples)
        results = aug.apply_transform(samples).samples
        self.assertEqual(results.shape, samples.shape)
        self.assertEqual(results.dtype, torch.float32)

