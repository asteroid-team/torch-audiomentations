import unittest

import numpy as np
import pytest
import torch
from numpy.testing import assert_almost_equal

from torch_audiomentations import PolarityInversion


class TestPolarityInversion(unittest.TestCase):
    def test_polarity_inversion(self):
        samples = np.array([[[1.0, 0.5, -0.25, -0.125, 0.0]]], dtype=np.float32)
        sample_rate = 16000

        augment = PolarityInversion(p=1.0, output_type="dict")
        inverted_samples = augment(
            samples=torch.from_numpy(samples), sample_rate=sample_rate
        ).samples.numpy()
        assert_almost_equal(
            inverted_samples,
            np.array([[[-1.0, -0.5, 0.25, 0.125, 0.0]]], dtype=np.float32),
        )
        self.assertEqual(inverted_samples.dtype, np.float32)

    def test_polarity_inversion_zero_probability(self):
        samples = np.array([[[1.0, 0.5, -0.25, -0.125, 0.0]]], dtype=np.float32)
        sample_rate = 16000

        augment = PolarityInversion(p=0.0, output_type="dict")
        processed_samples = augment(
            samples=torch.from_numpy(samples), sample_rate=sample_rate
        ).samples.numpy()
        assert_almost_equal(
            processed_samples,
            np.array([[[1.0, 0.5, -0.25, -0.125, 0.0]]], dtype=np.float32),
        )
        self.assertEqual(processed_samples.dtype, np.float32)

    def test_polarity_inversion_variability_within_batch(self):
        samples = np.array([[1.0, 0.5, 0.25, 0.125, 0.0]], dtype=np.float32)
        samples_batch = np.stack([samples] * 10000, axis=0)
        sample_rate = 16000

        augment = PolarityInversion(p=0.5, output_type="dict")
        processed_samples = augment(
            samples=torch.from_numpy(samples_batch), sample_rate=sample_rate
        ).samples.numpy()

        num_unprocessed_examples = 0
        num_processed_examples = 0
        for i in range(processed_samples.shape[0]):
            if np.sum(processed_samples[i]) > 0:
                num_unprocessed_examples += 1
            else:
                num_processed_examples += 1

        self.assertEqual(num_unprocessed_examples + num_processed_examples, 10000)

        print(num_processed_examples)
        self.assertGreater(num_processed_examples, 2000)
        self.assertLess(num_processed_examples, 8000)

    def test_polarity_inversion_multichannel(self):
        samples = np.array(
            [[[1.0, 0.5, -0.25, -0.125, 0.0]], [[1.0, 0.5, -0.25, -0.125, 0.0]]],
            dtype=np.float32,
        )
        sample_rate = 16000

        augment = PolarityInversion(p=1.0, output_type="dict")
        inverted_samples = augment(
            samples=torch.from_numpy(samples), sample_rate=sample_rate
        ).samples.numpy()
        assert_almost_equal(
            inverted_samples,
            np.array(
                [[[-1.0, -0.5, 0.25, 0.125, 0.0]], [[-1.0, -0.5, 0.25, 0.125, 0.0]]],
                dtype=np.float32,
            ),
        )
        self.assertEqual(inverted_samples.dtype, np.float32)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_polarity_inversion_cuda(self):
        samples = np.array([[[1.0, 0.5, -0.25, -0.125, 0.0]]], dtype=np.float32)
        sample_rate = 16000

        augment = PolarityInversion(p=1.0, output_type="dict").cuda()
        inverted_samples = (
            augment(samples=torch.from_numpy(samples).cuda(), sample_rate=sample_rate)
            .samples.cpu()
            .numpy()
        )
        assert_almost_equal(
            inverted_samples,
            np.array([[[-1.0, -0.5, 0.25, 0.125, 0.0]]], dtype=np.float32),
        )
        self.assertEqual(inverted_samples.dtype, np.float32)
