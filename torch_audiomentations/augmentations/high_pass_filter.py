import torch

from ..augmentations.low_pass_filter import LowPassFilter


class HighPassFilter(LowPassFilter):
    """
    Apply high-pass filtering to the input audio.
    """

    supports_multichannel = True
    requires_sample_rate = True

    def apply_transform(self, selected_samples: torch.Tensor, sample_rate: int = None):
        low_pass_filtered_samples = super().apply_transform(
            selected_samples.clone(), sample_rate
        )
        return selected_samples - low_pass_filtered_samples
