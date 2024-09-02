from typing import Optional
import warnings

import torch
from torch import Tensor


from ..core.transforms_interface import BaseWaveformTransform
from ..utils.object_dict import ObjectDict


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
        p_mode: Optional[str] = None,
        sample_rate: Optional[int] = None,
        target_rate: Optional[int] = None,
        output_type: Optional[str] = None,
    ):
        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        batch_size = samples.shape[0]
        num_channels = samples.shape[1]
        assert num_channels <= 255
        permutations = torch.zeros(
            (batch_size, num_channels), dtype=torch.int64, device=samples.device
        )
        for i in range(batch_size):
            permutations[i] = torch.randperm(num_channels, device=samples.device)
        self.transform_parameters["permutations"] = permutations

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        if samples.shape[1] == 1:
            warnings.warn(
                "Mono audio was passed to ShuffleChannels - there are no channels to shuffle."
                " The input will be returned unchanged."
            )
            return ObjectDict(
                samples=samples,
                sample_rate=sample_rate,
                targets=targets,
                target_rate=target_rate,
            )

        for i in range(samples.size(0)):
            samples[i] = samples[i, self.transform_parameters["permutations"][i]]
            if targets is not None:
                targets[i] = targets[i, self.transform_parameters["permutations"][i]]

        return ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )
