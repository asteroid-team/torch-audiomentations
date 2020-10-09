import torch
import unittest
from torch_audiomentations import ApplyBackgroundNoise, load_audio, calculate_rms
from .utils import TEST_FIXTURES_DIR


class TestApplyBackgroundNoise(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 16000
        self.batch_size = 32
        self.empty_input_audio = torch.empty(0)
        self.input_audio = torch.from_numpy(
            load_audio(
                TEST_FIXTURES_DIR / "acoustic_guitar_0.wav", sample_rate=self.sample_rate
            )
        ).unsqueeze(0)
        self.input_audios = torch.stack([self.input_audio] * self.batch_size).squeeze(1)
        self.bg_path = TEST_FIXTURES_DIR / "bg"
        self.bg_short_path = TEST_FIXTURES_DIR / "bg_short"
        self.bg_noise_transform_guaranteed = ApplyBackgroundNoise(self.bg_path, 20, p=1.0)
        self.bg_short_noise_transform_guaranteed = ApplyBackgroundNoise(
            self.bg_short_path, 20, p=1.0
        )
        self.bg_noise_transform_no_guarantee = ApplyBackgroundNoise(
            self.bg_path, 20, p=0.0
        )

    def test_background_noise_no_guarantee_with_single_tensor(self):
        mixed_input = self.bg_noise_transform_no_guarantee(
            self.input_audio, self.sample_rate
        )
        self.assertTrue(torch.equal(mixed_input, self.input_audio))
        self.assertEqual(mixed_input.size(0), self.input_audio.size(0))

    def test_background_noise_no_guarantee_with_empty_tensor(self):
        mixed_input = self.bg_noise_transform_no_guarantee(
            self.empty_input_audio, self.sample_rate
        )
        self.assertTrue(torch.equal(mixed_input, self.empty_input_audio))
        self.assertEqual(mixed_input.size(0), self.empty_input_audio.size(0))

    def test_background_noise_guaranteed_with_zero_length_samples(self):
        mixed_input = self.bg_noise_transform_guaranteed(
            self.empty_input_audio, self.sample_rate
        )
        self.assertTrue(torch.equal(mixed_input, self.empty_input_audio))
        self.assertEqual(mixed_input.size(0), self.empty_input_audio.size(0))

    def test_background_noise_guaranteed_with_single_tensor(self):
        mixed_input = self.bg_noise_transform_guaranteed(
            self.input_audio, self.sample_rate
        )
        self.assertFalse(torch.equal(mixed_input, self.input_audio))
        self.assertEqual(mixed_input.size(0), self.input_audio.size(0))
        self.assertEqual(mixed_input.size(1), self.input_audio.size(1))

    def test_background_noise_guaranteed_with_batched_tensor(self):
        mixed_inputs = self.bg_noise_transform_guaranteed(
            self.input_audios, self.sample_rate
        )
        self.assertFalse(torch.equal(mixed_inputs, self.input_audios))
        self.assertEqual(mixed_inputs.size(0), self.input_audios.size(0))
        self.assertEqual(mixed_inputs.size(1), self.input_audios.size(1))

    def test_background_short_noise_guaranteed_with_batched_tensor(self):
        mixed_input = self.bg_short_noise_transform_guaranteed(
            self.input_audio, self.sample_rate
        )
        self.assertFalse(torch.equal(mixed_input, self.input_audio))
        self.assertEqual(mixed_input.size(0), self.input_audio.size(0))
        self.assertEqual(mixed_input.size(1), self.input_audio.size(1))

    def test_varying_snr_within_batch(self):
        min_snr_in_db = 3
        max_snr_in_db = 30
        augment = ApplyBackgroundNoise(
            self.bg_path, min_snr_in_db=3, max_snr_in_db=30, p=1.0
        )
        augmented_audios = augment(self.input_audios, self.sample_rate)

        self.assertEqual(tuple(augmented_audios.shape), tuple(self.input_audios.shape))
        self.assertFalse(torch.equal(augmented_audios, self.input_audios))

        added_noises = augmented_audios - self.input_audios

        actual_snr_values = []
        for i in range(len(self.input_audios)):
            signal_rms = calculate_rms(self.input_audios[i])
            noise_rms = calculate_rms(added_noises[i])
            snr_in_db = 20 * torch.log10(signal_rms / noise_rms).item()
            self.assertGreaterEqual(snr_in_db, min_snr_in_db)
            self.assertLessEqual(snr_in_db, max_snr_in_db)

            actual_snr_values.append(snr_in_db)

        self.assertGreater(max(actual_snr_values) - min(actual_snr_values), 13.37)
