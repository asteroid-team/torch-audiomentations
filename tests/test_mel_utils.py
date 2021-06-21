import torch

from torch_audiomentations.utils.mel_scale import (
    convert_frequencies_to_mels,
    convert_mels_to_frequencies,
)


class TestMelUtils:
    def test_mel_utils_with_tensor_input(self):
        frequencies = torch.tensor([0.0, 1.0, 40.0, 400.0, 2000.0, 20000.0, 50000.0])
        mels = convert_frequencies_to_mels(frequencies)

        assert torch.allclose(
            mels,
            torch.tensor(
                [
                    0.0000e0,
                    1.6089e0,
                    6.2627e1,
                    5.0938e2,
                    1.5214e3,
                    3.8169e3,
                    4.8265e3,
                ]
            ),
            rtol=1e-4,
            atol=1e-3,
        )
        frequencies_again = convert_mels_to_frequencies(mels)
        assert torch.allclose(frequencies_again, frequencies, rtol=1e-5, atol=1e-3)

    def test_mel_utils_with_scalar_input(self):
        m = convert_frequencies_to_mels(torch.tensor(400.0))

        assert torch.allclose(
            m,
            torch.tensor(5.0938e2),
            rtol=1e-4,
            atol=1e-3,
        )
        f = convert_mels_to_frequencies(m)
        assert torch.allclose(f, torch.tensor(400.0), rtol=1e-5, atol=1e-3)
