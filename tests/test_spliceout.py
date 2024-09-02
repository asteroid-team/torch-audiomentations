import unittest
import numpy as np
import torch
import pytest

from torch_audiomentations.augmentations.splice_out import SpliceOut
from torch_audiomentations import Compose


class TestSpliceout(unittest.TestCase):
    def test_splice_out(self):
        audio_samples = torch.rand(size=(8, 1, 32000), dtype=torch.float32)
        augment = Compose(
            [
                SpliceOut(num_time_intervals=10, max_width=400, output_type="dict"),
            ],
            output_type="dict",
        )
        splice_out_samples = augment(
            samples=audio_samples, sample_rate=16000
        ).samples.numpy()

        assert splice_out_samples.dtype == np.float32

    def test_splice_out_odd_hann(self):
        audio_samples = torch.rand(size=(8, 1, 32000), dtype=torch.float32)
        augment = Compose(
            [
                SpliceOut(num_time_intervals=10, max_width=400, output_type="dict"),
            ],
            output_type="dict",
        )
        splice_out_samples = augment(
            samples=audio_samples, sample_rate=16100
        ).samples.numpy()

        assert splice_out_samples.dtype == np.float32

    def test_splice_out_per_batch(self):
        audio_samples = torch.rand(size=(8, 1, 32000), dtype=torch.float32)
        augment = Compose(
            [
                SpliceOut(
                    num_time_intervals=10,
                    max_width=400,
                    mode="per_batch",
                    p=1.0,
                    output_type="dict",
                ),
            ],
            output_type="dict",
        )
        splice_out_samples = augment(
            samples=audio_samples, sample_rate=16000
        ).samples.numpy()

        assert splice_out_samples.dtype == np.float32
        self.assertLess(splice_out_samples.sum(), audio_samples.numpy().sum())
        self.assertEqual(splice_out_samples.shape, audio_samples.shape)

    def test_splice_out_multichannel(self):
        audio_samples = torch.rand(size=(8, 2, 32000), dtype=torch.float32)
        augment = Compose(
            [
                SpliceOut(num_time_intervals=10, max_width=400, output_type="dict"),
            ],
            output_type="dict",
        )
        splice_out_samples = augment(
            samples=audio_samples, sample_rate=16000
        ).samples.numpy()

        assert splice_out_samples.dtype == np.float32
        self.assertLess(splice_out_samples.sum(), audio_samples.numpy().sum())
        self.assertEqual(splice_out_samples.shape, audio_samples.shape)

    @pytest.mark.skip(reason="This test fails and SpliceOut is not released yet")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_splice_out_cuda(self):
        audio_samples = (
            torch.rand(
                size=(8, 1, 32000), dtype=torch.float32, device=torch.device("cuda")
            )
            - 0.5
        )
        augment = Compose(
            [
                SpliceOut(num_time_intervals=10, max_width=400, output_type="dict"),
            ],
            output_type="dict",
        )
        splice_out_samples = (
            augment(samples=audio_samples, sample_rate=16000).samples.cpu().numpy()
        )

        assert splice_out_samples.dtype == np.float32
        self.assertLess(splice_out_samples.sum(), audio_samples.cpu().numpy().sum())
