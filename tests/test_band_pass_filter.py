import unittest

import numpy as np
import torch

from torch_audiomentations import BandPassFilter


class TestBandPassFilter(unittest.TestCase):
    def test_band_pass_filter(self):
        samples = np.array(
            [
                [[0.75, 0.5, -0.25, -0.125, 0.0], [0.65, 0.5, -0.25, -0.125, 0.0]],
                [[0.3, 0.5, -0.25, -0.125, 0.0], [0.9, 0.5, -0.25, -0.125, 0.0]],
                [[0.9, 0.5, -0.25, -1.06, 0.0], [0.9, 0.5, -0.25, -1.12, 0.0]],
            ],
            dtype=np.float32,
        )
        sample_rate = 16000

        augment = BandPassFilter(
            min_center_frequency=200,
            max_center_frequency=7500,
            min_bandwidth_fraction=0.25,
            max_bandwidth_fraction=1.5,
            p=1.0,
        )
        processed_samples = augment(
            samples=torch.from_numpy(samples), sample_rate=sample_rate
        ).numpy()
        self.assertEqual(processed_samples.shape, samples.shape)
        self.assertEqual(processed_samples.dtype, np.float32)
