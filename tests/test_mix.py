import unittest

import torch

from torch_audiomentations.augmentations.mix import Mix
from torch_audiomentations.utils.dsp import calculate_rms
from torch_audiomentations.utils.io import Audio
from .utils import TEST_FIXTURES_DIR


class TestMix(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 16000
        audio = Audio(self.sample_rate, mono=True)
        self.guitar = audio(TEST_FIXTURES_DIR / "acoustic_guitar_0.wav")[None]
        self.noise = audio(TEST_FIXTURES_DIR / "bg" / "bg.wav")[None]

        common_num_samples = min(self.guitar.shape[-1], self.noise.shape[-1])
        self.guitar = self.guitar[:, :, :common_num_samples]

        self.guitar_target = torch.zeros(
            (1, 1, common_num_samples // 7, 2), dtype=torch.int64
        )
        self.guitar_target[:, :, :, 0] = 1

        self.noise = self.noise[:, :, :common_num_samples]
        self.noise_target = torch.zeros(
            (1, 1, common_num_samples // 7, 2), dtype=torch.int64
        )
        self.noise_target[:, :, :, 1] = 1

        self.input_audios = torch.cat([self.guitar, self.noise], dim=0)
        self.input_targets = torch.cat([self.guitar_target, self.noise_target], dim=0)

    def test_varying_snr_within_batch(self):
        min_snr_in_db = 3
        max_snr_in_db = 30
        augment = Mix(
            min_snr_in_db=min_snr_in_db,
            max_snr_in_db=max_snr_in_db,
            p=1.0,
            output_type="dict",
        )
        mixed_audios = augment(self.input_audios, self.sample_rate).samples

        self.assertEqual(tuple(mixed_audios.shape), tuple(self.input_audios.shape))
        self.assertFalse(torch.equal(mixed_audios, self.input_audios))

        added_audios = mixed_audios - self.input_audios

        for i in range(len(self.input_audios)):
            signal_rms = calculate_rms(self.input_audios[i])
            added_rms = calculate_rms(added_audios[i])
            snr_in_db = 20 * torch.log10(signal_rms / added_rms).item()
            self.assertGreaterEqual(snr_in_db, min_snr_in_db)
            self.assertLessEqual(snr_in_db, max_snr_in_db)

    def test_targets_union(self):
        augment = Mix(p=1.0, mix_target="union", output_type="dict")
        mixtures = augment(
            samples=self.input_audios,
            sample_rate=self.sample_rate,
            targets=self.input_targets,
        )
        mixed_targets = mixtures.targets

        # check guitar target is still active in first (guitar) sample
        self.assertTrue(
            torch.equal(mixed_targets[0, :, :, 0], self.input_targets[0, :, :, 0])
        )
        # check noise target is still active in second (noise) sample
        self.assertTrue(
            torch.equal(mixed_targets[1, :, :, 1], self.input_targets[1, :, :, 1])
        )

    def test_targets_original(self):
        augment = Mix(p=1.0, mix_target="original", output_type="dict")
        mixtures = augment(
            samples=self.input_audios,
            sample_rate=self.sample_rate,
            targets=self.input_targets,
        )
        mixed_targets = mixtures.targets

        self.assertTrue(torch.equal(mixed_targets, self.input_targets))
