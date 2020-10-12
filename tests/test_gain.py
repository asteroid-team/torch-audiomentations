import unittest

import numpy as np
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
