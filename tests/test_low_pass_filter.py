import numpy as np
import torch

from torch_audiomentations import LowPassFilter


class TestLowPassFilter:
    def test_low_pass_filter(self):
        samples = np.array(
            [
                [[0.75, 0.5, -0.25, -0.125, 0.0], [0.65, 0.5, -0.25, -0.125, 0.0]],
                [[0.3, 0.5, -0.25, -0.125, 0.0], [0.9, 0.5, -0.25, -0.125, 0.0]],
                [[0.9, 0.5, -0.25, -1.06, 0.0], [0.9, 0.5, -0.25, -1.12, 0.0]],
            ],
            dtype=np.float32,
        )
        sample_rate = 16000

        augment = LowPassFilter(
            min_cutoff_freq=200, max_cutoff_freq=7000, p=1.0, output_type="dict"
        )
        processed_samples = augment(
            samples=torch.from_numpy(samples), sample_rate=sample_rate
        ).samples.numpy()
        assert processed_samples.shape == samples.shape
        assert processed_samples.dtype == np.float32

    def test_equal_cutoff_min_max(self):
        samples = np.array(
            [
                [[0.75, 0.5, -0.25, -0.125, 0.0], [0.65, 0.5, -0.25, -0.125, 0.0]],
                [[0.3, 0.5, -0.25, -0.125, 0.0], [0.9, 0.5, -0.25, -0.125, 0.0]],
                [[0.9, 0.5, -0.25, -1.06, 0.0], [0.9, 0.5, -0.25, -1.12, 0.0]],
            ],
            dtype=np.float32,
        )
        sample_rate = 16000

        augment = LowPassFilter(
            min_cutoff_freq=2000, max_cutoff_freq=2000, p=1.0, output_type="dict"
        )
        processed_samples = augment(
            samples=torch.from_numpy(samples), sample_rate=sample_rate
        ).samples.numpy()
        assert processed_samples.shape == samples.shape
        assert processed_samples.dtype == np.float32
