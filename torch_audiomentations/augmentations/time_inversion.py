import torch
from torch import Tensor
from typing import Optional

from ..core.transforms_interface import BaseWaveformTransform
from ..utils.object_dict import ObjectDict


class TimeInversion(BaseWaveformTransform):
    """
    Reverse (invert) the audio along the time axis similar to random flip of
    an image in the visual domain. This can be relevant in the context of audio
    classification. It was successfully applied in the paper
    AudioCLIP: Extending CLIP to Image, Text and Audio
    https://arxiv.org/pdf/2106.13043.pdf
    """

    supported_modes = {"per_batch", "per_example", "per_channel"}

    supports_multichannel = True
    requires_sample_rate = False

    supports_target = True
    requires_target = False

    def __init__(
        self,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
        target_rate: int = None,
        output_type: Optional[str] = None,
    ):
        """
        :param mode:
        :param p:
        :param p_mode:
        :param sample_rate:
        """
        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        # torch.flip() is supposed to be slower than np.flip()
        # An alternative is to use advanced indexing: https://github.com/pytorch/pytorch/issues/16424
        # reverse_index = torch.arange(selected_samples.size(-1) - 1, -1, -1).to(selected_samples.device)
        # transformed_samples = selected_samples[..., reverse_index]
        # return transformed_samples

        flipped_samples = torch.flip(samples, dims=(-1,))
        if targets is None:
            flipped_targets = targets
        else:
            flipped_targets = torch.flip(targets, dims=(-2,))

        return ObjectDict(
            samples=flipped_samples,
            sample_rate=sample_rate,
            targets=flipped_targets,
            target_rate=target_rate,
        )
