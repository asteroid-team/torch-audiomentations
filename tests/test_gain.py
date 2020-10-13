import unittest

import numpy as np
import pytest
import torch
from numpy.testing import assert_almost_equal

from torch_audiomentations.augmentations.gain import Gain
from torch_audiomentations.utils.dsp import (
    convert_decibels_to_amplitude_ratio,
    convert_amplitude_ratio_to_decibels,
)


class TestGain(unittest.TestCase):
    def test_gain(self):
        samples = np.array([[[1.0, 0.5, -0.25, -0.125, 0.0]]], dtype=np.float32)
        sample_rate = 16000

        augment = Gain(min_gain_in_db=-6.000001, max_gain_in_db=-6, p=1.0)
        processed_samples = augment(
            samples=torch.from_numpy(samples), sample_rate=sample_rate
        ).numpy()
        expected_factor = convert_decibels_to_amplitude_ratio(-6)
        assert_almost_equal(
            processed_samples,
            expected_factor
            * np.array([[[1.0, 0.5, -0.25, -0.125, 0.0]]], dtype=np.float32),
            decimal=6,
        )
        self.assertEqual(processed_samples.dtype, np.float32)

    def test_variability_within_batch(self):
        samples = np.array([1.0, 0.5, 0.25, 0.125, 0.01], dtype=np.float32)
        samples_batch = np.vstack([samples] * 10000)
        sample_rate = 16000

        augment = Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.5)
        processed_samples = augment(
            samples=torch.from_numpy(samples_batch), sample_rate=sample_rate
        ).numpy()
        self.assertEqual(processed_samples.dtype, np.float32)

        num_unprocessed_examples = 0
        num_processed_examples = 0
        actual_gains_in_db = []
        for i in range(processed_samples.shape[0]):
            if np.allclose(processed_samples[i], samples_batch[i]):
                num_unprocessed_examples += 1
            else:
                num_processed_examples += 1

                estimated_gain_factor = np.mean(processed_samples[i] / samples_batch[i])
                estimated_gain_factor_in_db = convert_amplitude_ratio_to_decibels(
                    torch.tensor(estimated_gain_factor)
                ).item()

                self.assertGreaterEqual(estimated_gain_factor_in_db, -6)
                self.assertLessEqual(estimated_gain_factor_in_db, 6)
                actual_gains_in_db.append(estimated_gain_factor_in_db)

        mean_gain_in_db = np.mean(actual_gains_in_db)
        self.assertGreater(mean_gain_in_db, -1)
        self.assertLess(mean_gain_in_db, 1)

        self.assertEqual(num_unprocessed_examples + num_processed_examples, 10000)
        self.assertGreater(num_processed_examples, 2000)
        self.assertLess(num_processed_examples, 8000)

    def test_reset_distribution(self):
        samples = np.array([1.0, 0.5, 0.25, 0.125, 0.01], dtype=np.float32)
        samples_batch = np.vstack([samples] * 10000)
        sample_rate = 16000

        augment = Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.5)
        # Change the parameters after init
        augment.min_gain_in_db = -18
        augment.max_gain_in_db = 3
        processed_samples = augment(
            samples=torch.from_numpy(samples_batch), sample_rate=sample_rate
        ).numpy()
        self.assertEqual(processed_samples.dtype, np.float32)

        actual_gains_in_db = []
        for i in range(processed_samples.shape[0]):
            if not np.allclose(processed_samples[i], samples_batch[i]):

                estimated_gain_factor = np.mean(processed_samples[i] / samples_batch[i])
                estimated_gain_factor_in_db = convert_amplitude_ratio_to_decibels(
                    torch.tensor(estimated_gain_factor)
                ).item()

                self.assertGreaterEqual(estimated_gain_factor_in_db, -18)
                self.assertLessEqual(estimated_gain_factor_in_db, 3)
                actual_gains_in_db.append(estimated_gain_factor_in_db)

        mean_gain_in_db = np.mean(actual_gains_in_db)
        self.assertGreater(mean_gain_in_db, (-18 + 3) / 2 - 1)
        self.assertLess(mean_gain_in_db, (-18 + 3) / 2 + 1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_cuda_reset_distribution(self):
        samples = np.array([1.0, 0.5, 0.25, 0.125, 0.01], dtype=np.float32)
        samples_batch = np.vstack([samples] * 10000)
        sample_rate = 16000

        augment = Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.5).cuda()
        # Change the parameters after init
        augment.min_gain_in_db = -18
        augment.max_gain_in_db = 3
        processed_samples = (
            augment(
                samples=torch.from_numpy(samples_batch).cuda(), sample_rate=sample_rate
            )
            .cpu()
            .numpy()
        )
        self.assertEqual(processed_samples.dtype, np.float32)

        actual_gains_in_db = []
        for i in range(processed_samples.shape[0]):
            if not np.allclose(processed_samples[i], samples_batch[i]):

                estimated_gain_factor = np.mean(processed_samples[i] / samples_batch[i])
                estimated_gain_factor_in_db = convert_amplitude_ratio_to_decibels(
                    torch.tensor(estimated_gain_factor)
                ).item()

                self.assertGreaterEqual(estimated_gain_factor_in_db, -18)
                self.assertLessEqual(estimated_gain_factor_in_db, 3)
                actual_gains_in_db.append(estimated_gain_factor_in_db)

        mean_gain_in_db = np.mean(actual_gains_in_db)
        self.assertGreater(mean_gain_in_db, (-18 + 3) / 2 - 1)
        self.assertLess(mean_gain_in_db, (-18 + 3) / 2 + 1)

    def test_invalid_distribution(self):
        with self.assertRaises(ValueError):
            Gain(min_gain_in_db=18, max_gain_in_db=-3, p=0.5)

        augment = Gain(min_gain_in_db=-6, max_gain_in_db=-3, p=1.0)
        # Change the parameters after init
        augment.min_gain_in_db = 18
        augment.max_gain_in_db = 3
        with self.assertRaises(ValueError):
            augment(torch.tensor([[[1.0, 0.5, 0.25, 0.125]]], dtype=torch.float32), 16000)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_gain_to_device_cuda(self):
        samples = np.array([[[1.0, 0.5, -0.25, -0.125, 0.0]]], dtype=np.float32)
        sample_rate = 16000

        cuda_device = torch.device("cuda")

        augment = Gain(min_gain_in_db=-6.000001, max_gain_in_db=-6, p=1.0).to(
            device=cuda_device
        )
        processed_samples = (
            augment(
                samples=torch.from_numpy(samples).to(device=cuda_device),
                sample_rate=sample_rate,
            )
            .cpu()
            .numpy()
        )
        expected_factor = convert_decibels_to_amplitude_ratio(-6)
        assert_almost_equal(
            processed_samples,
            expected_factor
            * np.array([[[1.0, 0.5, -0.25, -0.125, 0.0]]], dtype=np.float32),
            decimal=6,
        )
        self.assertEqual(processed_samples.dtype, np.float32)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_gain_cuda(self):
        samples = np.array([[[1.0, 0.5, -0.25, -0.125, 0.0]]], dtype=np.float32)
        sample_rate = 16000

        augment = Gain(min_gain_in_db=-6.000001, max_gain_in_db=-6, p=1.0).cuda()
        processed_samples = (
            augment(samples=torch.from_numpy(samples).cuda(), sample_rate=sample_rate)
            .cpu()
            .numpy()
        )
        expected_factor = convert_decibels_to_amplitude_ratio(-6)
        assert_almost_equal(
            processed_samples,
            expected_factor
            * np.array([[[1.0, 0.5, -0.25, -0.125, 0.0]]], dtype=np.float32),
            decimal=6,
        )
        self.assertEqual(processed_samples.dtype, np.float32)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_gain_cuda_cpu(self):
        samples = np.array([[[1.0, 0.5, -0.25, -0.125, 0.0]]], dtype=np.float32)
        sample_rate = 16000

        augment = Gain(min_gain_in_db=-6.000001, max_gain_in_db=-6, p=1.0).cuda().cpu()
        processed_samples = (
            augment(samples=torch.from_numpy(samples).cpu(), sample_rate=sample_rate)
            .cpu()
            .numpy()
        )
        expected_factor = convert_decibels_to_amplitude_ratio(-6)
        assert_almost_equal(
            processed_samples,
            expected_factor
            * np.array([[[1.0, 0.5, -0.25, -0.125, 0.0]]], dtype=np.float32),
            decimal=6,
        )
        self.assertEqual(processed_samples.dtype, np.float32)
