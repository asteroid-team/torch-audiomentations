import torch

from ..core.transforms_interface import BaseWaveformTransform


class TimeInversion(BaseWaveformTransform):
    """
    Reverse (invert) the audio along the time axis similar to random flip of
    an image in the visual domain. This can be relevant in the context of audio
    classification. It was successfully applied in the paper
    AudioCLIP: Extending CLIP to Image, Text and Audio
    https://arxiv.org/pdf/2106.13043.pdf
    """

    supports_multichannel = True
    requires_sample_rate = False

    def __init__(
        self,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
    ):
        """
        :param mode:
        :param p:
        :param p_mode:
        :param sample_rate:
        """
        super().__init__(mode, p, p_mode, sample_rate)

    def apply_transform(self, selected_samples: torch.Tensor, sample_rate: int = None):

        # torch.flip() is supposed to be slower than np.flip()
        # An alternative is to use advanced indexing: https://github.com/pytorch/pytorch/issues/16424
        # reverse_index = torch.arange(selected_samples.size(-1) - 1, -1, -1).to(selected_samples.device)
        # transformed_samples = selected_samples[..., reverse_index]
        # return transformed_samples
        return torch.flip(selected_samples, dims=(-1,))
