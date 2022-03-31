import numpy as np
import pytest
import torch

from torch_audiomentations.utils.io import Audio


class TestAudio:
    @pytest.mark.parametrize(
        "shape",
        [(1, 512), (1, 2, 512), (2, 1, 512)],
    )
    def test_rms_normalize(self, shape: tuple):
        samples = torch.rand(size=shape, dtype=torch.float32)
        normalized_samples = Audio.rms_normalize(samples)

        assert samples.shape == normalized_samples.shape

        normalized_rms = np.sqrt(np.mean(np.square(normalized_samples.numpy())))
        assert normalized_rms == pytest.approx(1.0)
