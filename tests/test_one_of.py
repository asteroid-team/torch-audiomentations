import unittest

import torch

from torch_audiomentations import PolarityInversion, PeakNormalization, Gain, OneOf
from torch_audiomentations.utils.object_dict import ObjectDict


class TestOneOf(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 16000
        self.audio = torch.randn(1, 1, 16000)

        self.transforms = [
            Gain(min_gain_in_db=-6.000001, max_gain_in_db=-2, p=1.0),
            PolarityInversion(p=1.0),
            PeakNormalization(p=1.0),
        ]

    def test_one_of_without_specifying_output_type(self):
        augment = OneOf(self.transforms)

        self.assertEqual(len(augment.transform_indexes), 0)  # no transforms applied yet
        output = augment(samples=self.audio, sample_rate=self.sample_rate)
        # This dtype should be torch.Tensor until we switch to ObjectDict by default
        assert type(output) == torch.Tensor

    def test_one_of_dict(self):
        augment = OneOf(self.transforms, output_type="dict")

        self.assertEqual(len(augment.transform_indexes), 0)  # no transforms applied yet
        output = augment(samples=self.audio, sample_rate=self.sample_rate)
        assert type(output) == ObjectDict
