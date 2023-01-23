import random
import unittest

import torch

from torch_audiomentations import AddColoredNoise
from torch_audiomentations.utils.io import Audio
from .utils import TEST_FIXTURES_DIR


class TestAddColoredNoise(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 16000
        self.audio = Audio(sample_rate=self.sample_rate)
        self.batch_size = 16
        self.empty_input_audio = torch.empty(0, 1, 16000)

        self.input_audio = self.audio(
            TEST_FIXTURES_DIR / "acoustic_guitar_0.wav"
        ).unsqueeze(0)

        self.input_audios = torch.cat([self.input_audio] * self.batch_size, dim=0)
        self.cl_noise_transform_guaranteed = AddColoredNoise(
            20, p=1.0, output_type="dict"
        )
        self.cl_noise_transform_no_guarantee = AddColoredNoise(
            20, p=0.0, output_type="dict"
        )

    def test_colored_noise_no_guarantee_with_single_tensor(self):
        mixed_input = self.cl_noise_transform_no_guarantee(
            self.input_audio, self.sample_rate
        ).samples
        self.assertTrue(torch.equal(mixed_input, self.input_audio))
        self.assertEqual(mixed_input.size(0), self.input_audio.size(0))

    def test_background_noise_no_guarantee_with_empty_tensor(self):
        with self.assertWarns(UserWarning) as warning_context_manager:
            mixed_input = self.cl_noise_transform_no_guarantee(
                self.empty_input_audio, self.sample_rate
            ).samples

        self.assertIn(
            "An empty samples tensor was passed", str(warning_context_manager.warning)
        )

        self.assertTrue(torch.equal(mixed_input, self.empty_input_audio))
        self.assertEqual(mixed_input.size(0), self.empty_input_audio.size(0))

    def test_colored_noise_guaranteed_with_zero_length_samples(self):

        with self.assertWarns(UserWarning) as warning_context_manager:
            mixed_input = self.cl_noise_transform_guaranteed(
                self.empty_input_audio, self.sample_rate
            ).samples

        self.assertIn(
            "An empty samples tensor was passed", str(warning_context_manager.warning)
        )

        self.assertTrue(torch.equal(mixed_input, self.empty_input_audio))
        self.assertEqual(mixed_input.size(0), self.empty_input_audio.size(0))

    def test_colored_noise_guaranteed_with_single_tensor(self):
        mixed_input = self.cl_noise_transform_guaranteed(
            self.input_audio, self.sample_rate
        ).samples
        self.assertFalse(torch.equal(mixed_input, self.input_audio))
        self.assertEqual(mixed_input.size(0), self.input_audio.size(0))
        self.assertEqual(mixed_input.size(1), self.input_audio.size(1))

    def test_colored_noise_guaranteed_with_batched_tensor(self):
        random.seed(42)
        mixed_inputs = self.cl_noise_transform_guaranteed(
            self.input_audios, self.sample_rate
        ).samples
        self.assertFalse(torch.equal(mixed_inputs, self.input_audios))
        self.assertEqual(mixed_inputs.size(0), self.input_audios.size(0))
        self.assertEqual(mixed_inputs.size(1), self.input_audios.size(1))

    def test_same_min_max_f_decay(self):
        random.seed(42)
        transform = AddColoredNoise(
            20, min_f_decay=1.0, max_f_decay=1.0, p=1.0, output_type="dict"
        )
        outputs = transform(self.input_audios, self.sample_rate).samples
        self.assertEqual(outputs.size(0), self.input_audios.size(0))
        self.assertEqual(outputs.size(1), self.input_audios.size(1))

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            AddColoredNoise(min_snr_in_db=30, max_snr_in_db=3, p=1.0, output_type="dict")
        with self.assertRaises(ValueError):
            AddColoredNoise(min_f_decay=2, max_f_decay=1, p=1.0, output_type="dict")
