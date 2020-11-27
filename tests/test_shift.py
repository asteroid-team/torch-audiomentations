import unittest

import numpy as np
import torch
from numpy.testing import assert_almost_equal

from torch_audiomentations import Shift


class TestShift(unittest.TestCase):
    def test_shift_by_1_sample_3dim(self):
        samples = torch.arange(4)[None, None].repeat(2, 2, 1)
        samples[1] += 1

        sample_rate = 16000

        augment = Shift(min_shift=1, max_shift=1, shift_unit="samples", p=1.0)
        processed_samples = augment(samples)

        assert_almost_equal(
            processed_samples,
            [
                [[3, 0, 1, 2], [3, 0, 1, 2]],
                [[4, 1, 2, 3], [4, 1, 2, 3]],
            ],
        )

        # self.assertEqual(processed_samples, np.int)

    def test_shift_by_1_sample_2dim(self):
        samples = torch.arange(4)[None].repeat(2, 1)

        sample_rate = 16000

        augment = Shift(min_shift=1, max_shift=1, shift_unit="samples", p=1.0)
        processed_samples = augment(samples)

        assert_almost_equal(
            processed_samples,
            [[3, 0, 1, 2], [3, 0, 1, 2]],
        )

    def test_shift_by_1_sample_without_rollover(self):
        samples = torch.arange(4)[None, None].repeat(2, 2, 1)
        samples[1] += 1
        sample_rate = 16000

        augment = Shift(
            min_shift=1,
            max_shift=1,
            shift_unit="samples",
            rollover=False,
            p=1.0,
        )

        processed_samples = augment(samples=samples)
        assert_almost_equal(
            processed_samples,
            [
                [[0, 0, 1, 2], [0, 0, 1, 2]],
                [[0, 1, 2, 3], [0, 1, 2, 3]],
            ],
        )

    def test_negative_shift_by_2_samples(self):
        samples = torch.arange(4)[None, None].repeat(2, 2, 1)
        samples[1] += 1
        sample_rate = 16000

        augment = Shift(
            min_shift=-2,
            max_shift=-2,
            shift_unit="samples",
            rollover=True,
            p=1.0,
        )

        processed_samples = augment(samples=samples)
        assert_almost_equal(
            processed_samples,
            [
                [[2, 3, 0, 1], [2, 3, 0, 1]],
                [[3, 4, 1, 2], [3, 4, 1, 2]],
            ],
        )

    def test_shift_by_fraction(self):
        samples = torch.arange(4)[None, None].repeat(2, 2, 1)
        samples[1] += 1
        sample_rate = 16000

        augment = Shift(
            min_shift=0.5,
            max_shift=0.5,
            shift_unit="fraction",
            rollover=True,
            p=1.0,
        )

        processed_samples = augment(samples=samples)
        assert_almost_equal(
            processed_samples,
            [
                [[2, 3, 0, 1], [2, 3, 0, 1]],
                [[3, 4, 1, 2], [3, 4, 1, 2]],
            ],
        )

    def test_shift_by_seconds(self):
        samples = torch.arange(4)[None, None].repeat(2, 2, 1)
        samples[1] += 1

        augment = Shift(max_shift=-2, shift_unit="seconds", p=1.0, sample_rate=1)
        processed_samples = augment(samples)
        
        assert_almost_equal(
            processed_samples,
            [
                [
                    [2, 3, 0, 1], 
                    [2, 3, 0, 1]
                ],
                [
                    [3, 4, 1, 2], 
                    [3, 4, 1, 2]
                ],
            ],
        )

    def test_shift_by_seconds_specify_sample_rate_in_both_init_and_forward(self):
        samples = torch.arange(4)[None, None].repeat(2, 2, 1)
        samples[1] += 1
        init_sample_rate = 1
        forward_sample_rate = 2

        augment = Shift(
            min_shift=1,
            max_shift=1,
            shift_unit="seconds",
            p=1.0,
            sample_rate=init_sample_rate,
        )
        # If sample_rate is specified in both __init__ and forward, then the latter will be used
        processed_samples = augment(samples, samples=forward_sample_rate)

        assert_almost_equal(
            processed_samples,
            [
                [
                    [0,0, 0, 1], 
                    [0,0, 0, 1], 
                ],
                [
                    [0, 0, 1, 2], 
                    [0, 0, 1, 2], 
                ],
            ],
        )

    def test_shift_by_seconds_without_specifying_sample_rate(self):
        samples = torch.arange(4)[None, None].repeat(2, 2, 1)
        samples[1] += 1

        augment = Shift(max_shift=-3, shift_unit="seconds", p=1.0)
        with self.assertRaises(RuntimeError):
            augment(samples)

        with self.assertRaises(RuntimeError):
            augment(samples, sample_rate=None)

    def test_variability_within_batch(self):
        torch.manual_seed(42)
        
        samples = torch.arange(4)[None, None].repeat(1000, 2, 1)
        samples[1] += 1
        breakpoint()
        augment = Shift(min_shift=-1, max_shift=1, shift_unit="samples", p=1.0)
        processed_samples = augment(samples)

        applied_shift_counts = {-1: 0, 0: 0, 1: 0}
        for i in range(samples.shape[0]):
            applied_shift = None
            for shift in range(0, augment.max_shift + 1):
                if np.array_equal(
                    np.roll(samples[i], shift, axis=-1), processed_samples[i]
                ):
                    applied_shift = shift
                    break
            self.assertIsNotNone(applied_shift)

            applied_shift_counts[applied_shift] += 1

        for shift in range(0, augment.max_shift + 1):
            self.assertGreater(applied_shift_counts[shift], 50)
