import unittest

import numpy as np
import torch
from numpy.testing import assert_array_equal

from torch_audiomentations import PolarityInversion, PeakNormalization, Gain
from torch_audiomentations import SomeOf
from torch_audiomentations.utils.object_dict import ObjectDict


class TestSomeOf(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 16000
        self.audio = torch.randn(1, 1, 16000)

        self.transforms = [
            Gain(min_gain_in_db=-6.000001, max_gain_in_db=-2, p=1.0),
            PolarityInversion(p=1.0),
            PeakNormalization(p=1.0),
        ]

    def test_some_of_without_specifying_output_type(self):
        augment = SomeOf(2, self.transforms)

        self.assertEqual(len(augment.transform_indexes), 0)  # no transforms applied yet
        output = augment(samples=self.audio, sample_rate=self.sample_rate)
        # This dtype should be torch.Tensor until we switch to ObjectDict by default
        assert type(output) == torch.Tensor
        self.assertEqual(len(augment.transform_indexes), 2)  # 2 transforms applied

    def test_some_of_dict(self):
        augment = SomeOf(2, self.transforms, output_type="dict")

        self.assertEqual(len(augment.transform_indexes), 0)  # no transforms applied yet
        output = augment(samples=self.audio, sample_rate=self.sample_rate)
        assert type(output) == ObjectDict
        self.assertEqual(len(augment.transform_indexes), 2)  # 2 transforms applied

    def test_some_of_with_p_zero(self):
        augment = SomeOf(2, self.transforms, p=0.0, output_type="dict")

        self.assertEqual(len(augment.transform_indexes), 0)  # no transforms applied yet
        processed_samples = augment(
            samples=self.audio, sample_rate=self.sample_rate
        ).samples
        self.assertEqual(len(augment.transform_indexes), 0)  # 0 transforms applied

    def test_some_of_tuple(self):
        augment = SomeOf((1, None), self.transforms, output_type="dict")

        self.assertEqual(len(augment.transform_indexes), 0)  # no transforms applied yet
        processed_samples = augment(
            samples=self.audio, sample_rate=self.sample_rate
        ).samples
        self.assertTrue(
            len(augment.transform_indexes) > 0
        )  # at least one transform applied

    def test_some_of_freeze_and_unfreeze_parameters(self):
        augment = SomeOf(2, self.transforms, output_type="dict")

        samples = np.array([[[1.0, 0.5, -0.25, -0.125, 0.0]]], dtype=np.float32)
        samples = torch.from_numpy(samples)

        self.assertEqual(len(augment.transform_indexes), 0)  # no transforms applied yet
        processed_samples1 = augment(
            samples=samples, sample_rate=self.sample_rate
        ).samples.numpy()
        transform_indexes1 = augment.transform_indexes
        self.assertEqual(len(augment.transform_indexes), 2)

        augment.freeze_parameters()

        processed_samples2 = augment(
            samples=samples, sample_rate=self.sample_rate
        ).samples.numpy()
        transform_indexes2 = augment.transform_indexes
        assert_array_equal(processed_samples1, processed_samples2)
        assert_array_equal(transform_indexes1, transform_indexes2)
