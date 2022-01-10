import random
import unittest
import torch

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

from torch_audiomentations import SomeOf
from torch_audiomentations import PolarityInversion, PeakNormalization, Gain


class TestSomeOf(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 16000
        self.audio = torch.randn(1, 1, 16000)

        self.transforms = [
            Gain(min_gain_in_db=-6.000001, max_gain_in_db=-2, p=1.0),
            PolarityInversion(p=1.0),
            PeakNormalization(p=1.0),
        ]

    def test_someof(self):
        augment = SomeOf(2, self.transforms)

        self.assertEqual(len(augment.transform_indexes), 0)  # no transforms applied yet
        processed_samples = augment(samples=self.audio, sample_rate=self.sample_rate)
        self.assertEqual(len(augment.transform_indexes), 2)  # 2 transforms applied

    def test_someof_with_p_zero(self):
        augment = SomeOf(2, self.transforms, p=0.0)

        self.assertEqual(len(augment.transform_indexes), 0)  # no transforms applied yet
        processed_samples = augment(samples=self.audio, sample_rate=self.sample_rate)
        self.assertEqual(len(augment.transform_indexes), 0)  # 0 transforms applied

    def test_someof_tuple(self):
        augment = SomeOf((1, None), self.transforms)

        self.assertEqual(len(augment.transform_indexes), 0)  # no transforms applied yet
        processed_samples = augment(samples=self.audio, sample_rate=self.sample_rate)
        self.assertTrue(len(augment.transform_indexes) > 0)  # at least one transform applied

    def test_someof_freeze_and_unfreeze_parameters(self):
        augment = SomeOf(2, self.transforms)

        samples = np.array([[[1.0, 0.5, -0.25, -0.125, 0.0]]], dtype=np.float32)
        samples = torch.from_numpy(samples)

        self.assertEqual(len(augment.transform_indexes), 0)  # no transforms applied yet
        processed_samples1 = augment(samples=samples, sample_rate=self.sample_rate).numpy()
        transform_indexes1 = augment.transform_indexes
        self.assertEqual(len(augment.transform_indexes), 2)

        augment.freeze_parameters()

        processed_samples2 = augment(samples=samples, sample_rate=self.sample_rate).numpy()
        transform_indexes2 = augment.transform_indexes
        assert_array_equal(processed_samples1, processed_samples2)
        assert_array_equal(transform_indexes1, transform_indexes2)
