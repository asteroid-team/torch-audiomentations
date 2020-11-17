import unittest

import numpy as np
import torch
from numpy.testing import assert_almost_equal

from torch_audiomentations import Shift


class TestShift(unittest.TestCase):
    def test_shift_by_1_sample_3dim(self):
        samples = np.array(
            [
                [[0.75, 0.5, -0.25, -0.125, 0.0], [0.9, 0.5, -0.25, -0.125, 0.0]],
                [[1, 0.5, 0.25, 0.125, 0.0], [1, 0.5, 0.25, 0.125, 0.06]],
            ],
            dtype=np.float32,
        )
        sample_rate = 16000

        augment = Shift(min_shift=1, max_shift=1, shift_unit="samples", p=1.0)
        processed_samples = augment(
            samples=torch.from_numpy(samples), sample_rate=sample_rate
        ).numpy()

        assert_almost_equal(
            processed_samples,
            np.array(
                [
                    [[0.0, 0.75, 0.5, -0.25, -0.125], [0.0, 0.9, 0.5, -0.25, -0.125]],
                    [[0.0, 1, 0.5, 0.25, 0.125], [0.06, 1, 0.5, 0.25, 0.125]],
                ],
                dtype=np.float32,
            ),
        )
        self.assertEqual(processed_samples.dtype, np.float32)

    def test_shift_by_1_sample_2dim(self):
        samples = np.array(
            [[0.75, 0.5, -0.25, -0.125, 0.77], [0.9, 0.5, -0.25, -0.125, 0.33]],
            dtype=np.float32,
        )
        sample_rate = 16000

        augment = Shift(min_shift=1, max_shift=1, shift_unit="samples", p=1.0)
        processed_samples = augment(
            samples=torch.from_numpy(samples), sample_rate=sample_rate
        ).numpy()

        assert_almost_equal(
            processed_samples,
            np.array(
                [[0.77, 0.75, 0.5, -0.25, -0.125], [0.33, 0.9, 0.5, -0.25, -0.125]],
                dtype=np.float32,
            ),
        )
        self.assertEqual(processed_samples.dtype, np.float32)

    def test_shift_by_1_sample_without_rollover(self):
        samples = np.array(
            [
                [[0.75, 0.5, -0.25, -0.125, 0.0], [0.9, 0.5, -0.25, -0.125, 0.0]],
                [[1, 0.5, 0.25, 0.125, 0.0], [1, 0.5, 0.25, 0.125, 0.06]],
            ],
            dtype=np.float32,
        )
        sample_rate = 16000

        augment = Shift(
            min_shift=1, max_shift=1, shift_unit="samples", rollover=False, p=1.0
        )
        processed_samples = augment(
            samples=torch.from_numpy(samples), sample_rate=sample_rate
        ).numpy()

        assert_almost_equal(
            processed_samples,
            np.array(
                [
                    [[0.0, 0.75, 0.5, -0.25, -0.125], [0.0, 0.9, 0.5, -0.25, -0.125]],
                    [[0.0, 1, 0.5, 0.25, 0.125], [0.00, 1, 0.5, 0.25, 0.125]],
                ],
                dtype=np.float32,
            ),
        )
        self.assertEqual(processed_samples.dtype, np.float32)

    def test_negative_shift_by_2_samples(self):
        samples = np.array(
            [
                [[0.75, 0.5, -0.25, -0.125, 0.0], [0.9, 0.5, -0.25, -0.125, 0.0]],
                [[1, 0.5, 0.25, 0.125, 0.0], [1, 0.5, 0.25, 0.125, 0.06]],
            ],
            dtype=np.float32,
        )
        sample_rate = 16000

        augment = Shift(min_shift=-2, max_shift=-2, shift_unit="samples", p=1.0)
        processed_samples = augment(
            samples=torch.from_numpy(samples), sample_rate=sample_rate
        ).numpy()

        assert_almost_equal(
            processed_samples,
            np.array(
                [
                    [[-0.25, -0.125, 0.0, 0.75, 0.5], [-0.25, -0.125, 0.0, 0.9, 0.5]],
                    [[0.25, 0.125, 0.0, 1, 0.5], [0.25, 0.125, 0.06, 1, 0.5]],
                ],
                dtype=np.float32,
            ),
        )
        self.assertEqual(processed_samples.dtype, np.float32)

    def test_shift_by_fraction(self):
        samples = np.array(
            [
                [[0.75, 0.5, -0.25, -0.125], [0.9, 0.5, -0.25, -0.125]],
                [[1, 0.5, 0.26, 0.125], [1, 0.5, 0.25, 0.125]],
            ],
            dtype=np.float32,
        )
        sample_rate = 16000

        augment = Shift(min_shift=0.5, max_shift=0.5, shift_unit="fraction", p=1.0)
        processed_samples = augment(
            samples=torch.from_numpy(samples), sample_rate=sample_rate
        ).numpy()

        assert_almost_equal(
            processed_samples,
            np.array(
                [
                    [[-0.25, -0.125, 0.75, 0.5], [-0.25, -0.125, 0.9, 0.5]],
                    [[0.26, 0.125, 1, 0.5], [0.25, 0.125, 1, 0.5]],
                ],
                dtype=np.float32,
            ),
        )
        self.assertEqual(processed_samples.dtype, np.float32)

    def test_shift_by_fraction_without_specifying_sample_rate(self):
        samples = np.array(
            [
                [[0.75, 0.5, -0.25, -0.125], [0.9, 0.5, -0.25, -0.125]],
                [[1, 0.5, 0.26, 0.125], [1, 0.5, 0.25, 0.125]],
            ],
            dtype=np.float32,
        )

        augment = Shift(min_shift=0.5, max_shift=0.5, shift_unit="fraction", p=1.0)
        processed_samples = augment(samples=torch.from_numpy(samples)).numpy()

        assert_almost_equal(
            processed_samples,
            np.array(
                [
                    [[-0.25, -0.125, 0.75, 0.5], [-0.25, -0.125, 0.9, 0.5]],
                    [[0.26, 0.125, 1, 0.5], [0.25, 0.125, 1, 0.5]],
                ],
                dtype=np.float32,
            ),
        )
        self.assertEqual(processed_samples.dtype, np.float32)

    def test_shift_by_seconds(self):
        samples = np.array(
            [
                [[0.75, 0.5, -0.25, -0.125], [0.9, 0.5, -0.25, -0.125]],
                [[1, 0.5, 0.26, 0.125], [1, 0.5, 0.25, 0.125]],
            ],
            dtype=np.float32,
        )
        sample_rate = 1

        augment = Shift(min_shift=-3, max_shift=-3, shift_unit="seconds", p=1.0)
        processed_samples = augment(
            samples=torch.from_numpy(samples), sample_rate=sample_rate
        ).numpy()

        assert_almost_equal(
            processed_samples,
            np.array(
                [
                    [[-0.125, 0.75, 0.5, -0.25], [-0.125, 0.9, 0.5, -0.25]],
                    [[0.125, 1, 0.5, 0.26], [0.125, 1, 0.5, 0.25]],
                ],
                dtype=np.float32,
            ),
        )
        self.assertEqual(processed_samples.dtype, np.float32)

    def test_shift_by_seconds_specify_sample_rate_in_init(self):
        samples = np.array(
            [
                [[0.75, 0.5, -0.25, -0.125], [0.9, 0.5, -0.25, -0.125]],
                [[1, 0.5, 0.26, 0.125], [1, 0.5, 0.25, 0.125]],
            ],
            dtype=np.float32,
        )
        sample_rate = 1

        augment = Shift(
            min_shift=-3,
            max_shift=-3,
            shift_unit="seconds",
            p=1.0,
            sample_rate=sample_rate,
        )
        processed_samples = augment(samples=torch.from_numpy(samples)).numpy()

        assert_almost_equal(
            processed_samples,
            np.array(
                [
                    [[-0.125, 0.75, 0.5, -0.25], [-0.125, 0.9, 0.5, -0.25]],
                    [[0.125, 1, 0.5, 0.26], [0.125, 1, 0.5, 0.25]],
                ],
                dtype=np.float32,
            ),
        )
        self.assertEqual(processed_samples.dtype, np.float32)

    def test_shift_by_seconds_without_specifying_sample_rate(self):
        samples = np.array(
            [
                [[0.75, 0.5, -0.25, -0.125], [0.9, 0.5, -0.25, -0.125]],
                [[1, 0.5, 0.26, 0.125], [1, 0.5, 0.25, 0.125]],
            ],
            dtype=np.float32,
        )

        augment = Shift(min_shift=-3, max_shift=-3, shift_unit="seconds", p=1.0)
        with self.assertRaises(RuntimeError):
            _ = augment(samples=torch.from_numpy(samples)).numpy()

        with self.assertRaises(RuntimeError):
            _ = augment(samples=torch.from_numpy(samples), sample_rate=None).numpy()

    def test_variability_within_batch(self):
        torch.manual_seed(42)

        samples = np.array(
            [[-0.25, -0.125, 0.0, 0.75, 0.5], [-0.25, -0.125, 0.0, 0.9, 0.5]],
            dtype=np.float32,
        )
        samples = np.stack([samples] * 1000, axis=0)
        sample_rate = 16000

        augment = Shift(min_shift=-1, max_shift=1, shift_unit="samples", p=1.0)
        processed_samples = augment(
            samples=torch.from_numpy(samples), sample_rate=sample_rate
        ).numpy()
        self.assertEqual(processed_samples.dtype, np.float32)

        applied_shift_counts = {-1: 0, 0: 0, 1: 0}
        for i in range(samples.shape[0]):
            applied_shift = None
            for shift in range(augment.min_shift, augment.max_shift + 1):
                if np.array_equal(
                    np.roll(samples[i], shift, axis=-1), processed_samples[i]
                ):
                    applied_shift = shift
                    break
            self.assertIsNotNone(applied_shift)

            applied_shift_counts[applied_shift] += 1

        for shift in range(augment.min_shift, augment.max_shift + 1):
            self.assertGreater(applied_shift_counts[shift], 50)
