import torch

from ..augmentations.band_pass_filter import BandPassFilter


class BandStopFilter(BandPassFilter):
    """
    Apply band-stop filtering to the input audio. Also known as notch filter.
    """

    supports_multichannel = True
    requires_sample_rate = True

    def __init__(
        self,
        min_center_frequency=200,
        max_center_frequency=4000,
        min_bandwidth_fraction=0.25,
        max_bandwidth_fraction=0.99,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
    ):
        """
        :param min_center_frequency: Minimum center frequency in hertz
        :param max_center_frequency: Maximum center frequency in hertz
        :param min_bandwidth_fraction: Minimum bandwidth in relative to center frequency
        :param max_bandwidth_fraction: Maximum bandwidth in relative to center frequency
        :param mode:
        :param p:
        :param p_mode:
        :param sample_rate:
        """

        super().__init__(
            min_center_frequency,
            max_center_frequency,
            min_bandwidth_fraction,
            max_bandwidth_fraction,
            mode, 
            p, 
            p_mode, 
            sample_rate,
        )

    def apply_transform(self, selected_samples: torch.Tensor, sample_rate: int = None):
        band_pass_filtered_samples = super().apply_transform(
            selected_samples.clone(), sample_rate
        )
        return selected_samples - band_pass_filtered_samples
