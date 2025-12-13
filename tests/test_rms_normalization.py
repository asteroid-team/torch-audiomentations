import unittest
import torch

from torch_audiomentations import RMSNormalization

# Allow some tolerance in dB comparisons due to numeric imprecision
THRESHOLD_DB = 1.0

class TestRMSNormalization(unittest.TestCase):
    def test_sine_wave_adjustment(self):
        """Transform should adjust a 440 Hz sine from -20 dBFS to -10 dBFS"""
        sample_rate = 16000
        t = torch.arange(0, sample_rate) / sample_rate
        wave = torch.sin(2 * torch.pi * 440 * t)

        # Scale to -20 dBFS
        current_rms = torch.sqrt(torch.mean(wave.pow(2)))
        target_amp = 10 ** (-20.0 / 20.0)
        wave = wave * (target_amp / current_rms)

        transform = RMSNormalization(-10.0)
        result = transform(wave.unsqueeze(0).unsqueeze(0), sample_rate)

        # Compute RMS of output
        out_rms = torch.sqrt(torch.mean(result.samples.pow(2))).item()

        # Manual RMS calculation for cross-check
        flattened = result.samples.flatten()
        total_power = sum(float(v) ** 2 for v in flattened)
        rms_manual = (total_power / flattened.numel()) ** 0.5

        # Convert to dB
        out_db = 20 * torch.log10(torch.tensor(out_rms)).item()

        # Verify vectorised and manual agree and that output is within tolerance
        assert abs(out_rms - rms_manual) < 1e-6
        assert abs(out_db + 10.0) < THRESHOLD_DB

    def test_silence_stays_zero(self):
        """Transform should leave a silent input unchanged"""
        zeros = torch.zeros(1, 1, 8000)
        result = RMSNormalization(-5.0)(zeros, 16000)
        self.assertTrue(torch.all(result.samples == 0))

    def test_stereo_equal_gain(self):
        """Left and right channels should be scaled by the same gain"""
        left  = torch.full((1, 1, 1000), 0.1)
        right = torch.full((1, 1, 1000), 0.2)
        stereo = torch.cat([left, right], dim = 1)

        transform = RMSNormalization(-6.0)
        result = transform(stereo, 16000)

        gain_left  = result.samples[0, 0, 0] / 0.1
        gain_right = result.samples[0, 1, 0] / 0.2

        # Allow small relative difference
        self.assertAlmostEqual(gain_left, gain_right, places = 3)
