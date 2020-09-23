import numpy as np
import os
import torch
import unittest
from numpy.testing import assert_almost_equal
from torch_audiomentations import load_audio
from torch_audiomentations import ImpulseResponse
from pathlib import Path


BASE_DIR = Path(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
TEST_FIXTURES_DIR = BASE_DIR / "test_fixtures"


class TestImpulseResponse(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 48000
        self.batch_size = 32
        self.input_audio = load_audio(os.path.join(TEST_FIXTURES_DIR, 'acoustic_guitar_0.wav'), self.sample_rate)
        self.input_audios = np.stack([self.input_audio] * self.batch_size)
        self.ir_path = os.path.join(TEST_FIXTURES_DIR, 'ir')
        self.ir_transform = ImpulseResponse(self.ir_path, p=1.0)

    def test_impulse_response_with_single_ndarray_input(self):
        mixed_input = self.ir_transform(self.input_audio, self.sample_rate)
        self.assertNotEqual(mixed_input.shape[-1], self.input_audio.shape[-1])

    def test_impulse_response_with_batched_ndarray_input(self):
        mixed_inputs = self.ir_transform(self.input_audios, self.sample_rate)
        self.assertEqual(mixed_inputs.shape[0], self.input_audios.shape[0])
        self.assertNotEqual(mixed_inputs.shape[-1], self.input_audios.shape[-1])

    def test_impulse_response_with_single_tensor_input(self):
        mixed_input = self.ir_transform(torch.from_numpy(self.input_audio), self.sample_rate)
        self.assertNotEqual(mixed_input.shape[-1], self.input_audio.shape[-1])

    def test_impulse_response_with_batched_tensor_input(self):
        mixed_inputs = self.ir_transform(torch.from_numpy(self.input_audios), self.sample_rate)
        self.assertEqual(mixed_inputs.shape[0], self.input_audios.shape[0])
        self.assertNotEqual(mixed_inputs.shape[-1], self.input_audios.shape[-1])

