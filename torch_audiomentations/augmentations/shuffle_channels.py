import typing
import warnings

import torch

from ..core.transforms_interface import BaseWaveformTransform


class ShuffleChannels(BaseWaveformTransform):
    """
    Given multichannel audio input (e.g. stereo), shuffle the channels, e.g. so left can become right and vice versa.
    This transform can help combat positional bias in machine learning models that input multichannel waveforms.

    If the input audio is mono, this transform does nothing except emit a warning.
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
        batch_size = selected_samples.shape[0]
        num_channels = selected_samples.shape[1]
        assert num_channels <= 255
        permutations = torch.zeros(
            (batch_size, num_channels), dtype=torch.int64, device=selected_samples.device
        )
        for i in range(batch_size):
            permutations[i] = torch.randperm(num_channels, device=selected_samples.device)
        self.transform_parameters["permutations"] = permutations

    def apply_transform(self, selected_samples, sample_rate: typing.Optional[int] = None):
        if selected_samples.shape[1] == 1:
            warnings.warn(
                "Mono audio was passed to ShuffleChannels - there are no channels to shuffle."
                " The input will be returned unchanged."
            )
            return selected_samples
        for i in range(selected_samples.size(0)):
            selected_samples[i] = selected_samples[
                i, self.transform_parameters["permutations"][i]
            ]
        return selected_samples
