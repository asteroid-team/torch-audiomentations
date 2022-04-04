import unittest

import numpy as np
import pytest
import torch
from numpy.testing import assert_almost_equal
from torch_audiomentations import PitchShift


def get_example():
    return (
        torch.rand(
            size=(8, 2, 32000),
            dtype=torch.float32,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        - 0.5
    )


class TestPitchShift(unittest.TestCase):
    def test_per_example_shift(self):
        samples = get_example()
        aug = PitchShift(sample_rate=16000, p=1, mode="per_example", output_type="dict")
        aug.randomize_parameters(samples)
        results = aug.apply_transform(samples).samples
        self.assertEqual(results.shape, samples.shape)

    def test_per_channel_shift(self):
        samples = get_example()
        aug = PitchShift(sample_rate=16000, p=1, mode="per_channel", output_type="dict")
        aug.randomize_parameters(samples)
        results = aug.apply_transform(samples).samples
        self.assertEqual(results.shape, samples.shape)

    def test_per_batch_shift(self):
        samples = get_example()
        aug = PitchShift(sample_rate=16000, p=1, mode="per_batch", output_type="dict")
        aug.randomize_parameters(samples)
        results = aug.apply_transform(samples).samples
        self.assertEqual(results.shape, samples.shape)

    def error_raised(self):
        error = False
        try:
            PitchShift(
                sample_rate=16000,
                p=1,
                mode="per_example",
                min_transpose_semitones=0.0,
                max_transpose_semitones=0.0,
                output_type="dict",
            )
        except ValueError:
            error = True
        if not error:
            raise ValueError("Invalid transpositions were not detected")
