import numpy as np
import pytest
import torch
from numpy.testing import assert_array_equal

from torch_audiomentations import ShuffleChannels
from torch_audiomentations.core.transforms_interface import ModeNotSupportedException


class TestShuffleChannels:
    def test_shuffle_mono(self):
        samples = torch.from_numpy(
            np.array([[[1.0, -1.0, 1.0, -1.0, 1.0]]], dtype=np.float32)
        )
        augment = ShuffleChannels(p=1.0, output_type="dict")

        with pytest.warns(UserWarning):
            processed_samples = augment(samples).samples

        assert_array_equal(samples.numpy(), processed_samples.numpy())

    @pytest.mark.parametrize(
        "device_name",
        [
            pytest.param("cpu"),
            pytest.param(
                "cuda",
                marks=pytest.mark.skip("Requires CUDA")
                if not torch.cuda.is_available()
                else [],
            ),
        ],
    )
    def test_variability_within_batch(self, device_name):
        device = torch.device(device_name)
        torch.manual_seed(42)

        samples = np.array(
            [[1.0, -1.0, 1.0, -1.0, 1.0], [0.1, -0.1, 0.1, -0.1, 0.1]], dtype=np.float32
        )
        samples = np.stack([samples] * 1000, axis=0)
        samples = torch.from_numpy(samples).to(device)

        augment = ShuffleChannels(p=1.0, output_type="dict")
        processed_samples = augment(samples).samples

        orders = {"original": 0, "swapped": 0}
        for i in range(processed_samples.shape[0]):
            if processed_samples[i, 0, 0] > 0.5:
                orders["original"] += 1
            else:
                orders["swapped"] += 1

        for order in orders:
            assert orders[order] > 50

    def test_unsupported_mode(self):
        with pytest.raises(ModeNotSupportedException):
            ShuffleChannels(mode="per_batch", p=1.0, output_type="dict")
