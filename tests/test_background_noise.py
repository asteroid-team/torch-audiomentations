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

from torch_audiomentations import AddBackgroundNoise
from torch_audiomentations.utils.dsp import calculate_rms
from .utils import TEST_FIXTURES_DIR
from torch_audiomentations.utils.io import Audio


class TestAddBackgroundNoise(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 16000
        self.batch_size = 16
        self.empty_input_audio = torch.empty(0, 1, 16000)
        # TODO: use utils.io.Audio

        audio = Audio(self.sample_rate, mono=True)
        self.input_audio = audio(TEST_FIXTURES_DIR / "acoustic_guitar_0.wav")[None]
        self.input_audios = torch.cat([self.input_audio] * self.batch_size, dim=0)

        self.bg_path = TEST_FIXTURES_DIR / "bg"
        self.bg_short_path = TEST_FIXTURES_DIR / "bg_short"
        self.bg_noise_transform_guaranteed = AddBackgroundNoise(
            self.bg_path, 20, p=1.0, output_type="dict"
        )
        self.bg_short_noise_transform_guaranteed = AddBackgroundNoise(
            self.bg_short_path, 20, p=1.0, output_type="dict"
        )
        self.bg_noise_transform_no_guarantee = AddBackgroundNoise(
            self.bg_path, 20, p=0.0, output_type="dict"
        )

    def test_background_noise_no_guarantee_with_single_tensor(self):
        mixed_input = self.bg_noise_transform_no_guarantee(
            self.input_audio, self.sample_rate
        ).samples
        self.assertTrue(torch.equal(mixed_input, self.input_audio))
        self.assertEqual(mixed_input.size(0), self.input_audio.size(0))

    def test_background_noise_no_guarantee_with_empty_tensor(self):
        with self.assertWarns(UserWarning) as warning_context_manager:
            mixed_input = self.bg_noise_transform_no_guarantee(
                self.empty_input_audio, self.sample_rate
            ).samples

        self.assertIn(
            "An empty samples tensor was passed", str(warning_context_manager.warning)
        )

        self.assertTrue(torch.equal(mixed_input, self.empty_input_audio))
        self.assertEqual(mixed_input.size(0), self.empty_input_audio.size(0))

    def test_background_noise_guaranteed_with_zero_length_samples(self):
        with self.assertWarns(UserWarning) as warning_context_manager:
            mixed_input = self.bg_noise_transform_guaranteed(
                self.empty_input_audio, self.sample_rate
            ).samples

        self.assertIn(
            "An empty samples tensor was passed", str(warning_context_manager.warning)
        )

        self.assertTrue(torch.equal(mixed_input, self.empty_input_audio))
        self.assertEqual(mixed_input.size(0), self.empty_input_audio.size(0))

    def test_background_noise_guaranteed_with_single_tensor(self):
        mixed_input = self.bg_noise_transform_guaranteed(
            self.input_audio, self.sample_rate
        ).samples
        self.assertFalse(torch.equal(mixed_input, self.input_audio))
        self.assertEqual(mixed_input.size(0), self.input_audio.size(0))
        self.assertEqual(mixed_input.size(1), self.input_audio.size(1))

    def test_background_noise_guaranteed_with_batched_tensor(self):
        random.seed(42)
        mixed_inputs = self.bg_noise_transform_guaranteed(
            self.input_audios, self.sample_rate
        ).samples
        self.assertFalse(torch.equal(mixed_inputs, self.input_audios))
        self.assertEqual(mixed_inputs.size(0), self.input_audios.size(0))
        self.assertEqual(mixed_inputs.size(1), self.input_audios.size(1))

    def test_background_short_noise_guaranteed_with_batched_tensor(self):
        mixed_input = self.bg_short_noise_transform_guaranteed(
            self.input_audio, self.sample_rate
        ).samples
        self.assertFalse(torch.equal(mixed_input, self.input_audio))
        self.assertEqual(mixed_input.size(0), self.input_audio.size(0))
        self.assertEqual(mixed_input.size(1), self.input_audio.size(1))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_background_short_noise_guaranteed_with_batched_cuda_tensor(self):
        input_audio_cuda = self.input_audio.cuda()
        mixed_input = self.bg_short_noise_transform_guaranteed(
            input_audio_cuda, self.sample_rate
        ).samples
        assert not torch.equal(mixed_input, input_audio_cuda)
        assert mixed_input.shape == input_audio_cuda.shape
        assert mixed_input.dtype == input_audio_cuda.dtype
        assert mixed_input.device == input_audio_cuda.device

    def test_varying_snr_within_batch(self):
        min_snr_in_db = 3
        max_snr_in_db = 30
        augment = AddBackgroundNoise(
            self.bg_path,
            min_snr_in_db=min_snr_in_db,
            max_snr_in_db=max_snr_in_db,
            p=1.0,
            output_type="dict",
        )
        augmented_audios = augment(self.input_audios, self.sample_rate).samples

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

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            augment = AddBackgroundNoise(
                self.bg_path, min_snr_in_db=30, max_snr_in_db=3, p=1.0, output_type="dict"
            )

    def test_min_equals_max(self):
        desired_snr = 3.0
        augment = AddBackgroundNoise(
            self.bg_path,
            min_snr_in_db=desired_snr,
            max_snr_in_db=desired_snr,
            p=1.0,
            output_type="dict",
        )
        augmented_audios = augment(self.input_audios, self.sample_rate).samples

        self.assertEqual(tuple(augmented_audios.shape), tuple(self.input_audios.shape))
        self.assertFalse(torch.equal(augmented_audios, self.input_audios))

        added_noises = augmented_audios - self.input_audios
        for i in range(len(self.input_audios)):
            signal_rms = calculate_rms(self.input_audios[i])
            noise_rms = calculate_rms(added_noises[i])
            snr_in_db = 20 * torch.log10(signal_rms / noise_rms).item()
            self.assertAlmostEqual(snr_in_db, desired_snr, places=5)

    def test_compatibility_of_resampled_length(self):
        random.seed(42)

        for _ in range(30):
            input_length = random.randint(1333, 1399)
            bg_length = random.randint(1333, 1399)
            input_sample_rate = random.randint(1000, 5000)
            bg_sample_rate = random.randint(1000, 5000)

            noise = np.random.uniform(
                low=-0.2,
                high=0.2,
                size=(bg_length,),
            ).astype(np.float32)
            tmp_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
            try:
                os.makedirs(tmp_dir)
                write(os.path.join(tmp_dir, "noise.wav"), rate=bg_sample_rate, data=noise)

                print(
                    f"input_length={input_length}, input_sample_rate={input_sample_rate},"
                    f" bg_length={bg_length}, bg_sample_rate={bg_sample_rate}"
                )
                input_audio = torch.randn(1, 1, input_length, dtype=torch.float32)
                transform = AddBackgroundNoise(
                    tmp_dir,
                    min_snr_in_db=4,
                    max_snr_in_db=6,
                    p=1.0,
                    sample_rate=input_sample_rate,
                    output_type="dict",
                )
                transform(input_audio)
            except Exception:
                raise
            finally:
                shutil.rmtree(tmp_dir)
