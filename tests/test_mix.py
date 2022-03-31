import os
import random
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io.wavfile import write

from torch_audiomentations import Mix
from torch_audiomentations.utils.dsp import calculate_rms
from torch_audiomentations.utils.file import load_audio
from .utils import TEST_FIXTURES_DIR


class TestMix(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 16000
        self.guitar = (
            torch.from_numpy(
                load_audio(
                    TEST_FIXTURES_DIR / "acoustic_guitar_0.wav",
                    sample_rate=self.sample_rate,
                )
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.noise = (
            torch.from_numpy(
                load_audio(
                    TEST_FIXTURES_DIR / "bg" / "bg.wav", sample_rate=self.sample_rate,
                )
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        common_num_samples = min(self.guitar.shape[-1], self.noise.shape[-1])
        self.guitar = self.guitar[:, :, :common_num_samples]
        self.noise = self.noise[:, :, :common_num_samples]
        self.input_audios = torch.cat([self.guitar, self.noise], dim=0)

    def test_varying_snr_within_batch(self):
        min_snr_in_db = 3
        max_snr_in_db = 30
        augment = Mix(min_snr_in_db=min_snr_in_db, max_snr_in_db=max_snr_in_db, p=1.0)
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

