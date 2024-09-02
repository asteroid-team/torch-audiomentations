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

        augment = Gain(
            min_gain_in_db=-6.000001, max_gain_in_db=-6, p=1.0, output_type="dict"
        )
        processed_samples = augment(
            samples=torch.from_numpy(samples), sample_rate=sample_rate
        ).samples.numpy()
        expected_factor = convert_decibels_to_amplitude_ratio(-6)
        assert_almost_equal(
            processed_samples,
            expected_factor
            * np.array([[[1.0, 0.5, -0.25, -0.125, 0.0]]], dtype=np.float32),
            decimal=6,
        )
        assert processed_samples.dtype == np.float32

    def test_gain_per_channel(self):
        samples = np.array(
            [[1.0, 0.5, 0.25, 0.125, 0.01], [0.95, 0.5, 0.25, 0.12, 0.011]],
            dtype=np.float32,
        )
        samples_batch = np.stack([samples] * 1000, axis=0)
        sample_rate = 16000

        augment = Gain(
            min_gain_in_db=-30.0,
            max_gain_in_db=0.0,
            mode="per_channel",
            p=0.5,
            output_type="dict",
        )
        processed_samples = augment(
            samples=torch.from_numpy(samples_batch), sample_rate=sample_rate
        ).samples.numpy()

        num_unprocessed_channels = 0
        num_processed_channels = 0
        perturbation_type_counter = {
            "zero_channels_changed": 0,
            "only_the_first_channel_changed": 0,
            "only_the_second_channel_changed": 0,
            "both_channels_changed_differently": 0,
            "both_channels_changed_equally": 0,
        }
        for i in range(processed_samples.shape[0]):
            num_perturbed_channels = 0
            est_gain_factors = []
            for chn_index in range(processed_samples.shape[1]):
                if np.allclose(
                    processed_samples[i, chn_index], samples_batch[i, chn_index]
                ):
                    num_unprocessed_channels += 1
                else:
                    num_processed_channels += 1
                    num_perturbed_channels += 1

                    estimated_gain_factors = (
                        processed_samples[i, chn_index] / samples_batch[i, chn_index]
                    )
                    self.assertAlmostEqual(
                        np.amin(estimated_gain_factors),
                        np.amax(estimated_gain_factors),
                        places=6,
                    )
                    estimated_gain_factor = np.mean(estimated_gain_factors)
                    estimated_gain_factor_in_db = convert_amplitude_ratio_to_decibels(
                        torch.tensor(estimated_gain_factor)
                    ).item()
                    est_gain_factors.append(estimated_gain_factor_in_db)

                    self.assertGreaterEqual(estimated_gain_factor_in_db, -30)
                    self.assertLessEqual(estimated_gain_factor_in_db, 0)

            if num_perturbed_channels == 0:
                perturbation_type_counter["zero_channels_changed"] += 1
            elif num_perturbed_channels == 1:
                if np.allclose(processed_samples[i, 0], samples_batch[i, 0]):
                    perturbation_type_counter["only_the_first_channel_changed"] += 1
                else:
                    perturbation_type_counter["only_the_second_channel_changed"] += 1
            elif num_perturbed_channels == 2:
                if np.allclose(est_gain_factors[0], est_gain_factors[1]):
                    # This should be very unlikely
                    perturbation_type_counter["both_channels_changed_equally"] += 1
                else:
                    perturbation_type_counter["both_channels_changed_differently"] += 1

        self.assertGreater(perturbation_type_counter["zero_channels_changed"], 100)
        self.assertLess(perturbation_type_counter["zero_channels_changed"], 500)
        self.assertGreater(
            perturbation_type_counter["only_the_first_channel_changed"], 100
        )
        self.assertLess(perturbation_type_counter["only_the_first_channel_changed"], 500)
        self.assertGreater(
            perturbation_type_counter["only_the_second_channel_changed"], 100
        )
        self.assertLess(perturbation_type_counter["only_the_second_channel_changed"], 500)
        self.assertLess(perturbation_type_counter["both_channels_changed_equally"], 10)
        self.assertGreater(
            perturbation_type_counter["both_channels_changed_differently"], 100
        )
        self.assertLess(
            perturbation_type_counter["both_channels_changed_differently"], 500
        )

        self.assertEqual(num_unprocessed_channels + num_processed_channels, 1000 * 2)
        self.assertGreater(num_processed_channels, 200 * 2)
        self.assertLess(num_processed_channels, 800 * 2)

        self.assertEqual(processed_samples.dtype, np.float32)

    def test_gain_per_channel_with_p_mode_per_batch(self):
        samples = np.array(
            [[1.0, 0.5, 0.25, 0.125, 0.01], [0.95, 0.5, 0.25, 0.12, 0.011]],
            dtype=np.float32,
        )
        samples_batch = np.stack([samples] * 100, axis=0)
        sample_rate = 16000

        augment = Gain(
            min_gain_in_db=-30.0,
            max_gain_in_db=-1.0,
            mode="per_channel",
            p=0.5,
            p_mode="per_batch",
            output_type="dict",
        )
        num_processed_batches = 0
        for i in range(100):
            processed_samples = augment(
                samples=torch.from_numpy(samples_batch), sample_rate=sample_rate
            ).samples.numpy()

            if np.allclose(processed_samples, samples_batch):
                continue
            else:
                num_processed_batches += 1

            num_unprocessed_channels = 0
            num_processed_channels = 0
            perturbation_type_counter = {
                "zero_channels_changed": 0,
                "only_the_first_channel_changed": 0,
                "only_the_second_channel_changed": 0,
                "both_channels_changed_differently": 0,
                "both_channels_changed_equally": 0,
            }
            for i in range(processed_samples.shape[0]):
                num_perturbed_channels = 0
                est_gain_factors = []
                for chn_index in range(processed_samples.shape[1]):
                    if np.allclose(
                        processed_samples[i, chn_index], samples_batch[i, chn_index]
                    ):
                        num_unprocessed_channels += 1
                    else:
                        num_processed_channels += 1
                        num_perturbed_channels += 1

                        estimated_gain_factors = (
                            processed_samples[i, chn_index] / samples_batch[i, chn_index]
                        )
                        self.assertAlmostEqual(
                            np.amin(estimated_gain_factors),
                            np.amax(estimated_gain_factors),
                            places=6,
                        )
                        estimated_gain_factor = np.mean(estimated_gain_factors)
                        estimated_gain_factor_in_db = convert_amplitude_ratio_to_decibels(
                            torch.tensor(estimated_gain_factor)
                        ).item()
                        est_gain_factors.append(estimated_gain_factor_in_db)

                        self.assertGreaterEqual(estimated_gain_factor_in_db, -30)
                        self.assertLessEqual(estimated_gain_factor_in_db, 0)

                if num_perturbed_channels == 0:
                    perturbation_type_counter["zero_channels_changed"] += 1
                elif num_perturbed_channels == 1:
                    if np.allclose(processed_samples[i, 0], samples_batch[i, 0]):
                        perturbation_type_counter["only_the_first_channel_changed"] += 1
                    else:
                        perturbation_type_counter["only_the_second_channel_changed"] += 1
                elif num_perturbed_channels == 2:
                    if np.allclose(est_gain_factors[0], est_gain_factors[1]):
                        # This should be very unlikely
                        perturbation_type_counter["both_channels_changed_equally"] += 1
                    else:
                        perturbation_type_counter[
                            "both_channels_changed_differently"
                        ] += 1

            self.assertEqual(perturbation_type_counter["zero_channels_changed"], 0)
            self.assertLess(
                perturbation_type_counter["only_the_first_channel_changed"], 2
            )
            self.assertLess(
                perturbation_type_counter["only_the_second_channel_changed"], 2
            )
            self.assertLess(perturbation_type_counter["both_channels_changed_equally"], 2)
            self.assertGreater(
                perturbation_type_counter["both_channels_changed_differently"], 10
            )

            self.assertEqual(num_processed_channels, 100 * 2)
            self.assertEqual(processed_samples.dtype, np.float32)

        self.assertGreater(num_processed_batches, 10)
        self.assertLess(num_processed_batches, 90)

    def test_gain_per_channel_with_p_mode_per_example(self):
        samples = np.array(
            [[1.0, 0.5, 0.25, 0.125, 0.01], [0.95, 0.5, 0.25, 0.12, 0.011]],
            dtype=np.float32,
        )
        samples_batch = np.stack([samples] * 10000, axis=0)

        augment = Gain(
            min_gain_in_db=-30.0,
            max_gain_in_db=-1.0,
            mode="per_channel",
            p=0.5,
            p_mode="per_example",
            output_type="dict",
        )
        processed_samples = augment(
            samples=torch.from_numpy(samples_batch), sample_rate=None
        ).samples.numpy()

        num_unprocessed_examples = 0
        num_processed_examples = 0
        perturbation_type_counter = {
            "zero_channels_changed": 0,
            "only_the_first_channel_changed": 0,
            "only_the_second_channel_changed": 0,
            "both_channels_changed_differently": 0,
            "both_channels_changed_equally": 0,
        }
        for i in range(processed_samples.shape[0]):
            num_perturbed_channels = 0
            est_gain_factors = []

            if np.allclose(processed_samples[i], samples_batch[i]):
                num_unprocessed_examples += 1
            else:
                num_processed_examples += 1

            for chn_index in range(processed_samples.shape[1]):
                if not np.allclose(
                    processed_samples[i, chn_index], samples_batch[i, chn_index]
                ):
                    num_perturbed_channels += 1

                    estimated_gain_factors = (
                        processed_samples[i, chn_index] / samples_batch[i, chn_index]
                    )
                    self.assertAlmostEqual(
                        np.amin(estimated_gain_factors),
                        np.amax(estimated_gain_factors),
                        places=6,
                    )
                    estimated_gain_factor = np.mean(estimated_gain_factors)
                    estimated_gain_factor_in_db = convert_amplitude_ratio_to_decibels(
                        torch.tensor(estimated_gain_factor)
                    ).item()
                    est_gain_factors.append(estimated_gain_factor_in_db)

                    self.assertGreaterEqual(estimated_gain_factor_in_db, -30)
                    self.assertLessEqual(estimated_gain_factor_in_db, 0)

            if num_perturbed_channels == 0:
                perturbation_type_counter["zero_channels_changed"] += 1
            elif num_perturbed_channels == 1:
                if np.allclose(processed_samples[i, 0], samples_batch[i, 0]):
                    perturbation_type_counter["only_the_first_channel_changed"] += 1
                else:
                    perturbation_type_counter["only_the_second_channel_changed"] += 1
            elif num_perturbed_channels == 2:
                if np.allclose(est_gain_factors[0], est_gain_factors[1]):
                    # This should be very unlikely
                    perturbation_type_counter["both_channels_changed_equally"] += 1
                else:
                    perturbation_type_counter["both_channels_changed_differently"] += 1

        self.assertGreater(perturbation_type_counter["zero_channels_changed"], 2000)
        self.assertLess(perturbation_type_counter["zero_channels_changed"], 8000)

        # We allow both these two to be 1 due to numerical accuracy. Ideally they should be 0.
        self.assertLessEqual(
            perturbation_type_counter["only_the_first_channel_changed"], 1
        )
        self.assertLessEqual(
            perturbation_type_counter["only_the_second_channel_changed"], 1
        )

        self.assertLess(perturbation_type_counter["both_channels_changed_equally"], 100)
        self.assertGreater(
            perturbation_type_counter["both_channels_changed_differently"], 2000
        )
        self.assertLess(
            perturbation_type_counter["both_channels_changed_differently"], 8000
        )

        self.assertEqual(num_unprocessed_examples + num_processed_examples, 10000)
        self.assertGreater(num_processed_examples, 2000)
        self.assertLess(num_processed_examples, 8000)

        self.assertEqual(processed_samples.dtype, np.float32)

    def test_gain_per_batch(self):
        samples = np.array(
            [[1.0, 0.5, 0.25, 0.125, 0.01], [0.95, 0.5, 0.25, 0.12, 0.011]],
            dtype=np.float32,
        )
        samples_batch = np.stack([samples] * 4, axis=0)
        sample_rate = 16000

        augment = Gain(
            min_gain_in_db=-30.0,
            max_gain_in_db=0.0,
            mode="per_batch",
            p=0.5,
            output_type="dict",
        )
        num_unprocessed_batches = 0
        num_processed_batches = 0
        for i in range(1000):
            processed_samples = augment(
                samples=torch.from_numpy(samples_batch), sample_rate=sample_rate
            ).samples.numpy()

            estimated_gain_factors = processed_samples / samples_batch
            self.assertAlmostEqual(
                np.amin(estimated_gain_factors), np.amax(estimated_gain_factors), places=6
            )
            if np.allclose(processed_samples, samples_batch):
                num_unprocessed_batches += 1
            else:
                num_processed_batches += 1

        self.assertGreater(num_processed_batches, 200)
        self.assertGreater(num_unprocessed_batches, 200)

    def test_eval(self):
        samples = np.array([[[1.0, 0.5, -0.25, -0.125, 0.0]]], dtype=np.float32)
        sample_rate = 16000

        augment = Gain(min_gain_in_db=-15, max_gain_in_db=5, p=1.0, output_type="dict")
        augment.eval()

        processed_samples = augment(
            samples=torch.from_numpy(samples), sample_rate=sample_rate
        ).samples.numpy()

        np.testing.assert_array_equal(samples, processed_samples)

    def test_variability_within_batch(self):
        samples = np.array([[1.0, 0.5, 0.25, 0.125, 0.01]], dtype=np.float32)
        samples_batch = np.stack([samples] * 10000, axis=0)
        sample_rate = 16000

        augment = Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.5, output_type="dict")
        processed_samples = augment(
            samples=torch.from_numpy(samples_batch), sample_rate=sample_rate
        ).samples.numpy()
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

    def test_variability_within_batch_with_p_mode_per_batch(self):
        samples = np.array([[1.0, 0.5, 0.25, 0.125, 0.01]], dtype=np.float32)
        samples_batch = np.stack([samples] * 100, axis=0)
        sample_rate = 16000

        augment = Gain(
            min_gain_in_db=-6,
            max_gain_in_db=6,
            p=0.5,
            p_mode="per_batch",
            output_type="dict",
        )

        num_processed_batches = 0
        for _ in range(100):
            processed_samples = augment(
                samples=torch.from_numpy(samples_batch), sample_rate=sample_rate
            ).samples.numpy()
            self.assertEqual(processed_samples.dtype, np.float32)

            if np.allclose(processed_samples, samples_batch):
                continue
            else:
                num_processed_batches += 1

            num_unprocessed_examples = 0
            num_processed_examples = 0
            actual_gains_in_db = []
            for i in range(processed_samples.shape[0]):
                if np.allclose(processed_samples[i], samples_batch[i]):
                    num_unprocessed_examples += 1
                else:
                    num_processed_examples += 1

                    estimated_gain_factor = np.mean(
                        processed_samples[i] / samples_batch[i]
                    )
                    estimated_gain_factor_in_db = convert_amplitude_ratio_to_decibels(
                        torch.tensor(estimated_gain_factor)
                    ).item()

                    self.assertGreaterEqual(estimated_gain_factor_in_db, -6)
                    self.assertLessEqual(estimated_gain_factor_in_db, 6)
                    actual_gains_in_db.append(estimated_gain_factor_in_db)

            mean_gain_in_db = np.mean(actual_gains_in_db)
            self.assertGreater(mean_gain_in_db, -1.5)
            self.assertLess(mean_gain_in_db, 1.5)

            # Should be 0 and 100, but I give some slack due to possible numerical issues
            self.assertLessEqual(num_unprocessed_examples, 1)
            self.assertGreaterEqual(num_processed_examples, 99)

        self.assertGreater(num_processed_batches, 10)
        self.assertLess(num_processed_batches, 90)

    def test_reset_distribution(self):
        samples = np.array([[1.0, 0.5, 0.25, 0.125, 0.01]], dtype=np.float32)
        samples_batch = np.stack([samples] * 10000, axis=0)
        sample_rate = 16000

        augment = Gain(
            min_gain_in_db=-6,
            max_gain_in_db=6,
            p=0.5,
            sample_rate=sample_rate,
            output_type="dict",
        )
        # Change the parameters after init
        augment.min_gain_in_db = -18
        augment.max_gain_in_db = 3
        processed_samples = augment(
            samples=torch.from_numpy(samples_batch)
        ).samples.numpy()
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
        samples = np.array([[1.0, 0.5, 0.25, 0.125, 0.01]], dtype=np.float32)
        samples_batch = np.stack([samples] * 10000, axis=0)
        sample_rate = 16000

        augment = Gain(
            min_gain_in_db=-6, max_gain_in_db=6, p=0.5, output_type="dict"
        ).cuda()
        # Change the parameters after init
        augment.min_gain_in_db = -18
        augment.max_gain_in_db = 3
        processed_samples = (
            augment(
                samples=torch.from_numpy(samples_batch).cuda(), sample_rate=sample_rate
            )
            .samples.cpu()
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

        augment = Gain(min_gain_in_db=-6, max_gain_in_db=-3, p=1.0, output_type="dict")
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

        augment = Gain(
            min_gain_in_db=-6.000001, max_gain_in_db=-6, p=1.0, output_type="dict"
        ).to(device=cuda_device)
        processed_samples = (
            augment(
                samples=torch.from_numpy(samples).to(device=cuda_device),
                sample_rate=sample_rate,
            )
            .samples.cpu()
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

        augment = Gain(
            min_gain_in_db=-6.000001, max_gain_in_db=-6, p=1.0, output_type="dict"
        ).cuda()
        processed_samples = (
            augment(samples=torch.from_numpy(samples).cuda(), sample_rate=sample_rate)
            .samples.cpu()
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

        augment = (
            Gain(min_gain_in_db=-6.000001, max_gain_in_db=-6, p=1.0, output_type="dict")
            .cuda()
            .cpu()
        )
        processed_samples = (
            augment(samples=torch.from_numpy(samples).cpu(), sample_rate=sample_rate)
            .samples.cpu()
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
