import librosa
import torch
from numpy.testing import assert_almost_equal
from scipy.signal import convolve as scipy_convolve

from tests.utils import TEST_FIXTURES_DIR
from torch_audiomentations.utils.convolution import convolve as torch_convolve


class TestConvolution:
    def test_convolve(self):
        sample_rate = 16000

        file_path = TEST_FIXTURES_DIR / "acoustic_guitar_0.wav"
        samples, _ = librosa.load(file_path, sr=sample_rate)
        ir_samples, _ = librosa.load(
            TEST_FIXTURES_DIR / "ir" / "impulse_response_0.wav", sr=sample_rate
        )

        expected_output = scipy_convolve(samples, ir_samples)
        actual_output = torch_convolve(
            torch.from_numpy(samples), torch.from_numpy(ir_samples)
        ).numpy()

        assert_almost_equal(actual_output, expected_output, decimal=6)
