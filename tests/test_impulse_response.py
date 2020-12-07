import os
import unittest

import torch

from torch_audiomentations import ApplyImpulseResponse
from torch_audiomentations.core.transforms_interface import (
    MultichannelAudioNotSupportedException,
)
from torch_audiomentations.utils.file import load_audio
from .utils import TEST_FIXTURES_DIR


class TestApplyImpulseResponse(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 16000
        self.batch_size = 32
        self.empty_input_audio = torch.empty(0)
        self.input_audio = (
            torch.from_numpy(
                load_audio(
                    os.path.join(TEST_FIXTURES_DIR, "acoustic_guitar_0.wav"),
                    self.sample_rate,
                )
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.input_audios = torch.cat([self.input_audio] * self.batch_size, dim=0)
        self.ir_path = os.path.join(TEST_FIXTURES_DIR, "ir")
        self.ir_transform_guaranteed = ApplyImpulseResponse(self.ir_path, p=1.0)
        self.ir_transform_no_guarantee = ApplyImpulseResponse(self.ir_path, p=0.0)

    def test_impulse_response_guaranteed_with_single_tensor_input(self):
        mixed_input = self.ir_transform_guaranteed(self.input_audio, self.sample_rate)
        self.assertEqual(mixed_input.size(0), self.input_audio.size(0))
        self.assertEqual(mixed_input.size(-1), self.input_audio.size(-1))
        self.assertFalse(torch.equal(mixed_input, self.input_audio))

    def test_impulse_response_guaranteed_with_batched_tensor_input(self):
        mixed_inputs = self.ir_transform_guaranteed(self.input_audios, self.sample_rate)
        self.assertEqual(mixed_inputs.size(0), self.input_audios.size(0))
        self.assertEqual(mixed_inputs.size(-1), self.input_audios.size(-1))

        self.assertFalse(torch.equal(mixed_inputs, self.input_audios))

    def test_impulse_response_no_guarantee_with_single_tensor_input(self):
        mixed_input = self.ir_transform_no_guarantee(self.input_audio, self.sample_rate)
        self.assertEqual(mixed_input.size(-1), self.input_audio.size(-1))

    def test_impulse_response_no_guarantee_with_batched_tensor_input(self):
        mixed_inputs = self.ir_transform_no_guarantee(self.input_audios, self.sample_rate)
        self.assertEqual(mixed_inputs.size(0), self.input_audios.size(0))
        self.assertEqual(mixed_inputs.size(-1), self.input_audios.size(-1))

    def test_impulse_response_guaranteed_with_zero_length_samples(self):
        with self.assertWarns(UserWarning) as warning_context_manager:
            mixed_inputs = self.ir_transform_guaranteed(
                self.empty_input_audio, self.sample_rate
            )

        self.assertIn(
            "An empty samples tensor was passed", str(warning_context_manager.warning)
        )

        self.assertTrue(torch.equal(mixed_inputs, self.empty_input_audio))

    def test_multichannel_not_supported(self):
        # TODO: Remove this test when it supports multichannel audio
        audio = torch.rand(4, 2, 16000, dtype=torch.float32)
        with self.assertRaises(MultichannelAudioNotSupportedException):
            ApplyImpulseResponse(self.ir_path, p=1.0)(audio)
