import torch
import unittest
from torch_audiomentations import ApplyBackgroundNoise, load_audio
from .utils import TEST_FIXTURES_DIR


class TestApplyBackgroundNoise(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 16000
        self.batch_size = 2
        self.empty_input_audio = torch.empty(0)
        self.input_audio = torch.from_numpy(
            load_audio(TEST_FIXTURES_DIR / "acoustic_guitar_0.wav", self.sample_rate)
        ).unsqueeze(0)
        self.input_audios = torch.stack([self.input_audio] * self.batch_size).squeeze(1)
        self.bg_path = TEST_FIXTURES_DIR / "bg"
        self.bg_noise_transform_guaranteed = ApplyBackgroundNoise(self.bg_path, 20, p=1.0)
        self.bg_noise_transform_no_guarantee = ApplyBackgroundNoise(self.bg_path, 20, p=0.0)

    def test_background_noise_no_guarantee_with_single_tensor(self):
        mixed_input = self.bg_noise_transform_no_guarantee(self.input_audio, self.sample_rate)
        self.assertTrue(torch.equal(mixed_input, self.input_audio))
        self.assertEqual(mixed_input.size(0), self.input_audio.size(0))

    def test_background_noise_no_guarantee_with_empty_tensor(self):
        mixed_input = self.bg_noise_transform_no_guarantee(self.empty_input_audio, self.sample_rate)
        self.assertTrue(torch.equal(mixed_input, self.empty_input_audio))
        self.assertEqual(mixed_input.size(0), self.empty_input_audio.size(0))

    def test_background_noise_guaranteed_with_zero_length_samples(self):
        mixed_input = self.bg_noise_transform_guaranteed(self.empty_input_audio, self.sample_rate)
        self.assertTrue(torch.equal(mixed_input, self.empty_input_audio))
        self.assertEqual(mixed_input.size(0), self.empty_input_audio.size(0))

    def test_background_noise_guaranteed_with_single_tensor(self):
        mixed_input = self.bg_noise_transform_guaranteed(self.input_audio, self.sample_rate)
        self.assertFalse(torch.equal(mixed_input, self.input_audio))
        self.assertEqual(mixed_input.size(0), self.input_audio.size(0))
        self.assertEqual(mixed_input.size(1), self.input_audio.size(1))

    def test_background_noise_guaranteed_with_batched_tensor(self):
        mixed_inputs = self.bg_noise_transform_guaranteed(self.input_audios, self.sample_rate)
        self.assertFalse(torch.equal(mixed_inputs, self.input_audios))
        self.assertEqual(mixed_inputs.size(0), self.input_audios.size(0))
        self.assertEqual(mixed_inputs.size(1), self.input_audios.size(1))
