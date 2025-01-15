import torch
from numpy.testing import assert_almost_equal
from scipy.signal import convolve as scipy_convolve

from tests.utils import TEST_FIXTURES_DIR
from torch_audiomentations.utils.convolution import convolve as torch_convolve
from torch_audiomentations.utils.io import Audio


class TestConvolution:
    def test_convolve(self):
        sample_rate = 16000

        file_path = TEST_FIXTURES_DIR / "acoustic_guitar_0.wav"
        audio = Audio(sample_rate, mono=True)
        samples = audio(file_path).numpy()
        ir_samples = audio(TEST_FIXTURES_DIR / "ir" / "impulse_response_0.wav").numpy()

        expected_output = scipy_convolve(samples, ir_samples)
        actual_output = torch_convolve(
            torch.from_numpy(samples), torch.from_numpy(ir_samples)
        ).numpy()

        assert_almost_equal(actual_output, expected_output, decimal=6)
