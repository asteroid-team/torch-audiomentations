import unittest

import numpy as np
import pytest
import torch
from numpy.testing import assert_almost_equal, assert_equal

from torch_audiomentations.augmentations.peak_normalization import PeakNormalization


class TestPeakNormalization(unittest.TestCase):
    def test_apply_to_all(self):
        samples = np.array(
            [
                [[0.75, 0.5, -0.25, -0.125, 0.0]],
                [[0.9, 0.5, -0.25, -0.125, 0.0]],
                [[0.9, 0.5, -0.25, -1.12, 0.0]],
            ],
            dtype=np.float32,
        )
        sample_rate = 16000

        augment = PeakNormalization(p=1.0, output_type="dict")
        processed_samples = augment(
            samples=torch.from_numpy(samples), sample_rate=sample_rate
        ).samples.numpy()

        assert_almost_equal(
            processed_samples,
            np.array(
                [
                    [[0.75 / 0.75, 0.5 / 0.75, -0.25 / 0.75, -0.125 / 0.75, 0.0 / 0.75]],
                    [[0.9 / 0.9, 0.5 / 0.9, -0.25 / 0.9, -0.125 / 0.9, 0.0 / 0.9]],
                    [[0.9 / 1.12, 0.5 / 1.12, -0.25 / 1.12, -1.12 / 1.12, 0.0 / 1.12]],
                ],
                dtype=np.float32,
            ),
        )
        self.assertEqual(processed_samples.dtype, np.float32)

    def test_apply_to_only_too_loud_sounds(self):
        samples = np.array(
            [
                [[0.75, 0.5, -0.25, -0.125, 0.0]],
                [[0.9, 0.5, -0.25, -0.125, 0.0]],
                [[0.9, 0.5, -0.25, -1.12, 0.0]],
            ],
            dtype=np.float32,
        )
        sample_rate = 16000

        augment = PeakNormalization(
            apply_to="only_too_loud_sounds", p=1.0, output_type="dict"
        )
        processed_samples = augment(
            samples=torch.from_numpy(samples), sample_rate=sample_rate
        ).samples.numpy()

        assert_almost_equal(
            processed_samples,
            np.array(
                [
                    [[0.75, 0.5, -0.25, -0.125, 0.0]],
                    [[0.9, 0.5, -0.25, -0.125, 0.0]],
                    [[0.9 / 1.12, 0.5 / 1.12, -0.25 / 1.12, -1.12 / 1.12, 0.0 / 1.12]],
                ],
                dtype=np.float32,
            ),
        )
        self.assertEqual(processed_samples.dtype, np.float32)

    def test_digital_silence_in_batch(self):
        """Check that there is no division by zero in case of digital silence (all zeros)."""
        samples = np.array(
            [[[0.75, 0.5, -0.25, -0.125, 0.0]], [[0.0, 0.0, 0.0, 0.0, 0.0]]],
            dtype=np.float32,
        )
        sample_rate = 16000

        augment = PeakNormalization(p=1.0, output_type="dict")
        processed_samples = augment(
            samples=torch.from_numpy(samples), sample_rate=sample_rate
        ).samples.numpy()

        assert_almost_equal(
            processed_samples,
            np.array(
                [
                    [[0.75 / 0.75, 0.5 / 0.75, -0.25 / 0.75, -0.125 / 0.75, 0.0 / 0.75]],
                    [[0.0, 0.0, 0.0, 0.0, 0.0]],
                ],
                dtype=np.float32,
            ),
        )
        self.assertEqual(processed_samples.dtype, np.float32)

    def test_only_digital_silence(self):
        """Check that an exception is not thrown is selector is all False."""
        samples = np.array(
            [[[0.0, 0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=np.float32
        )
        sample_rate = 16000

        augment = PeakNormalization(p=1.0, output_type="dict")
        processed_samples = augment(
            samples=torch.from_numpy(samples), sample_rate=sample_rate
        ).samples.numpy()

        assert_almost_equal(
            processed_samples,
            np.array(
                [[[0.0, 0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0, 0.0]]],
                dtype=np.float32,
            ),
        )
        self.assertEqual(processed_samples.dtype, np.float32)

    def test_never_apply(self):
        samples = np.array(
            [
                [[0.75, 0.5, -0.25, -0.125, 0.0]],
                [[0.9, 0.5, -0.25, -0.125, 0.0]],
                [[0.9, 0.5, -0.25, -1.12, 0.0]],
            ],
            dtype=np.float32,
        )
        sample_rate = 16000

        augment = PeakNormalization(p=0.0, output_type="dict")
        processed_samples = augment(
            samples=torch.from_numpy(samples), sample_rate=sample_rate
        ).samples.numpy()

        assert_equal(
            processed_samples,
            np.array(
                [
                    [[0.75, 0.5, -0.25, -0.125, 0.0]],
                    [[0.9, 0.5, -0.25, -0.125, 0.0]],
                    [[0.9, 0.5, -0.25, -1.12, 0.0]],
                ],
                dtype=np.float32,
            ),
        )
        self.assertEqual(processed_samples.dtype, np.float32)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_apply_to_all_cuda(self):
        samples = np.array(
            [
                [[0.75, 0.5, -0.25, -0.125, 0.0]],
                [[0.9, 0.5, -0.25, -0.125, 0.0]],
                [[0.9, 0.5, -0.25, -1.12, 0.0]],
            ],
            dtype=np.float32,
        )
        sample_rate = 16000

        augment = PeakNormalization(p=1.0, output_type="dict")
        processed_samples = (
            augment(samples=torch.from_numpy(samples).cuda(), sample_rate=sample_rate)
            .samples.cpu()
            .numpy()
        )

        assert_almost_equal(
            processed_samples,
            np.array(
                [
                    [[0.75 / 0.75, 0.5 / 0.75, -0.25 / 0.75, -0.125 / 0.75, 0.0 / 0.75]],
                    [[0.9 / 0.9, 0.5 / 0.9, -0.25 / 0.9, -0.125 / 0.9, 0.0 / 0.9]],
                    [[0.9 / 1.12, 0.5 / 1.12, -0.25 / 1.12, -1.12 / 1.12, 0.0 / 1.12]],
                ],
                dtype=np.float32,
            ),
        )
        self.assertEqual(processed_samples.dtype, np.float32)

    def test_variability_within_batch(self):
        samples = np.array([[0.75, 0.5, 0.25, 0.125, 0.01]], dtype=np.float32)
        samples_batch = np.stack([samples] * 1337, axis=0)
        sample_rate = 16000

        augment = PeakNormalization(p=0.5, output_type="dict")
        processed_samples = augment(
            samples=torch.from_numpy(samples_batch), sample_rate=sample_rate
        ).samples.numpy()
        self.assertEqual(processed_samples.dtype, np.float32)

        num_unprocessed_examples = 0
        num_processed_examples = 0
        for i in range(processed_samples.shape[0]):
            if np.allclose(processed_samples[i], samples_batch[i]):
                num_unprocessed_examples += 1
            else:
                num_processed_examples += 1

        self.assertEqual(num_unprocessed_examples + num_processed_examples, 1337)
        self.assertGreater(num_processed_examples, 0.2 * 1337)
        self.assertLess(num_processed_examples, 0.8 * 1337)

    def test_freeze_parameters(self):
        samples1 = np.array(
            [[[0.9, 0.5, -0.25, -0.125, 0.0]], [[0.9, 0.5, -0.25, -1.12, 0.0]]],
            dtype=np.float32,
        )
        samples2 = np.array(
            [[[0.1, -0.2, -0.35, -0.625, 2.0]], [[0.2, 0.9, -0.05, -0.12, 0.0]]],
            dtype=np.float32,
        )
        sample_rate = 16000

        augment = PeakNormalization(p=1.0, output_type="dict")
        _ = augment(
            samples=torch.from_numpy(samples1), sample_rate=sample_rate
        ).samples.numpy()
        augment.freeze_parameters()
        processed_samples2 = augment(
            samples=torch.from_numpy(samples2), sample_rate=sample_rate
        ).samples.numpy()
        augment.unfreeze_parameters()

        assert_almost_equal(
            processed_samples2,
            np.array(
                [
                    [[0.1 / 0.9, -0.2 / 0.9, -0.35 / 0.9, -0.625 / 0.9, 2.0 / 0.9]],
                    [[0.2 / 1.12, 0.9 / 1.12, -0.05 / 1.12, -0.12 / 1.12, 0.0 / 1.12]],
                ],
                dtype=np.float32,
            ),
        )
        self.assertEqual(processed_samples2.dtype, np.float32)

    def test_stereo_sound(self):
        samples = np.array(
            [
                [[0.75, 0.5, -0.25, -0.125, 0.0], [0.65, 0.5, -0.25, -0.125, 0.0]],
                [[0.3, 0.5, -0.25, -0.125, 0.0], [0.9, 0.5, -0.25, -0.125, 0.0]],
                [[0.9, 0.5, -0.25, -1.06, 0.0], [0.9, 0.5, -0.25, -1.12, 0.0]],
            ],
            dtype=np.float32,
        )
        sample_rate = 16000

        augment = PeakNormalization(apply_to="all", p=1.0, output_type="dict")
        processed_samples = augment(
            samples=torch.from_numpy(samples), sample_rate=sample_rate
        ).samples.numpy()

        assert_almost_equal(
            processed_samples,
            np.array(
                [
                    np.array(
                        [[0.75, 0.5, -0.25, -0.125, 0.0], [0.65, 0.5, -0.25, -0.125, 0.0]]
                    )
                    / 0.75,
                    np.array(
                        [[0.3, 0.5, -0.25, -0.125, 0.0], [0.9, 0.5, -0.25, -0.125, 0.0]]
                    )
                    / 0.9,
                    np.array(
                        [[0.9, 0.5, -0.25, -1.06, 0.0], [0.9, 0.5, -0.25, -1.12, 0.0]]
                    )
                    / 1.12,
                ],
                dtype=np.float32,
            ),
        )
        self.assertEqual(processed_samples.dtype, np.float32)
