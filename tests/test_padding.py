import unittest
import numpy as np
import torch
from numpy.testing import assert_almost_equal
import pytest

from torch_audiomentations.augmentations.padding import Padding


class TestPadding(unittest.TestCase):
    def test_padding_end(self):
        audio_samples = torch.rand(size=(2, 2, 32000), dtype=torch.float32)
        augment = Padding(
            min_fraction=0.2,
            max_fraction=0.5,
            pad_section="end",
            p=1.0,
            output_type="dict",
        )
        padded_samples = augment(audio_samples).samples

        self.assertEqual(audio_samples.shape, padded_samples.shape)
        assert_almost_equal(padded_samples[..., -6400:].numpy(), np.zeros((2, 2, 6400)))

    def test_padding_start(self):
        audio_samples = torch.rand(size=(2, 2, 32000), dtype=torch.float32)
        augment = Padding(
            min_fraction=0.2,
            max_fraction=0.5,
            pad_section="start",
            p=1.0,
            output_type="dict",
        )
        padded_samples = augment(audio_samples).samples

        self.assertEqual(audio_samples.shape, padded_samples.shape)
        assert_almost_equal(padded_samples[..., :6400].numpy(), np.zeros((2, 2, 6400)))

    def test_padding_zero(self):
        audio_samples = torch.rand(size=(2, 2, 32000), dtype=torch.float32)
        augment = Padding(min_fraction=0.2, max_fraction=0.5, p=0.0, output_type="dict")
        padded_samples = augment(audio_samples).samples

        self.assertEqual(audio_samples.shape, padded_samples.shape)
        assert_almost_equal(audio_samples.numpy(), padded_samples.numpy())

    def test_padding_perexample(self):
        audio_samples = torch.rand(size=(10, 2, 32000), dtype=torch.float32)
        augment = Padding(
            min_fraction=0.2,
            max_fraction=0.5,
            pad_section="start",
            p=0.5,
            mode="per_example",
            p_mode="per_example",
            output_type="dict",
        )

        padded_samples = augment(audio_samples).samples.numpy()
        num_unprocessed_examples = 0.0
        num_processed_examples = 0.0
        for i, sample in enumerate(padded_samples):
            if np.allclose(audio_samples[i], sample):
                num_unprocessed_examples += 1
            else:
                num_processed_examples += 1

        self.assertLess(padded_samples.sum(), audio_samples.numpy().sum())

    def test_padding_perchannel(self):
        audio_samples = torch.rand(size=(10, 2, 32000), dtype=torch.float32)
        augment = Padding(
            min_fraction=0.2,
            max_fraction=0.5,
            pad_section="start",
            p=0.5,
            mode="per_channel",
            p_mode="per_channel",
            output_type="dict",
        )

        padded_samples = augment(audio_samples).samples.numpy()
        num_unprocessed_examples = 0.0
        num_processed_examples = 0.0
        for i, sample in enumerate(padded_samples):
            if np.allclose(audio_samples[i], sample):
                num_unprocessed_examples += 1
            else:
                num_processed_examples += 1

        self.assertLess(padded_samples.sum(), audio_samples.numpy().sum())

    def test_padding_variability_perexample(self):
        audio_samples = torch.rand(size=(10, 2, 32000), dtype=torch.float32)
        augment = Padding(
            min_fraction=0.2,
            max_fraction=0.5,
            pad_section="start",
            p=0.5,
            mode="per_example",
            p_mode="per_example",
            output_type="dict",
        )

        padded_samples = augment(audio_samples).samples.numpy()
        num_unprocessed_examples = 0.0
        num_processed_examples = 0.0
        for i, sample in enumerate(padded_samples):
            if np.allclose(audio_samples[i], sample):
                num_unprocessed_examples += 1
            else:
                num_processed_examples += 1

        self.assertEqual(num_processed_examples + num_unprocessed_examples, 10)
        self.assertGreater(num_processed_examples, 2)
        self.assertLess(num_unprocessed_examples, 8)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_padding_cuda(self):
        audio_samples = torch.rand(
            size=(2, 2, 32000), dtype=torch.float32, device=torch.device("cuda")
        )
        augment = Padding(min_fraction=0.2, max_fraction=0.5, p=1.0, output_type="dict")
        padded_samples = augment(audio_samples).samples

        self.assertEqual(audio_samples.shape, padded_samples.shape)
        assert_almost_equal(
            padded_samples[..., -6400:].cpu().numpy(), np.zeros((2, 2, 6400))
        )
