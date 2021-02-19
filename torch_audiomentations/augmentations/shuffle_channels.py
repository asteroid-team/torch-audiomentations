import typing

import torch

from ..core.transforms_interface import BaseWaveformTransform


class ShuffleChannels(BaseWaveformTransform):
    """
    Shuffle the audio channels if the audio is multichannel (e.g. stereo).
    This transform can help combat positional bias.

    This transform does nothing except raise a warning if the input audio is mono.
    """

    supports_multichannel = True
    requires_sample_rate = False
    supported_modes = {"per_example"}

    def __init__(
        self,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: typing.Optional[str] = None,
        sample_rate: typing.Optional[int] = None,
    ):
        super().__init__(mode, p, p_mode, sample_rate)

    def randomize_parameters(
        self, selected_samples, sample_rate: typing.Optional[int] = None
    ):
        batch_size = selected_samples.size(0)
        num_channels = selected_samples.shape[1]
        assert num_channels <= 255
        permutations = torch.zeros(
            (batch_size, num_channels), dtype=torch.int64, device=selected_samples.device
        )
        for i in range(batch_size):
            permutations[i] = torch.randperm(num_channels, device=selected_samples.device)
        self.transform_parameters["permutations"] = permutations

    def apply_transform(self, selected_samples, sample_rate: typing.Optional[int] = None):
        for i in range(selected_samples.size(0)):
            selected_samples[i] = selected_samples[
                i, self.transform_parameters["permutations"][i]
            ]
        return selected_samples
