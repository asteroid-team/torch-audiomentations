import julius
import torch

from ..core.transforms_interface import BaseWaveformTransform
from ..utils.mel_scale import convert_frequencies_to_mels, convert_mels_to_frequencies


class BandPassFilter(BaseWaveformTransform):
    """
    Apply band-pass filtering to the input audio.
    """

    supports_multichannel = True
    requires_sample_rate = True

    def __init__(
        self,
        low_cutoff_freq_range: tuple = (100, 600),
        high_cutoff_freq_range: tuple = (3000, 8000),
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
    ):
        """
        :param min_cutoff_freq: Minimum cutoff frequency in hertz
        :param max_cutoff_freq: Maximum cutoff frequency in hertz
        :param mode:
        :param p:
        :param p_mode:
        :param sample_rate:
        """
        super().__init__(mode, p, p_mode, sample_rate)

        self.low_cutoff_freq_range = low_cutoff_freq_range
        self.high_cutoff_freq_range = high_cutoff_freq_range
        if self.low_cutoff_freq_range[1] > self.high_cutoff_freq_range[0]:
            raise ValueError("min_cutoff_freq must not be greater than max_cutoff_freq")

    def randomize_parameters(
        self, selected_samples: torch.Tensor, sample_rate: int = None
    ):
        """
        :params selected_samples: (batch_size, num_channels, num_samples)
        """
        batch_size, _, num_samples = selected_samples.shape

        # Sample frequencies uniformly in mel space, then convert back to frequency
        def get_dist(cutoff_freq_range):
            dist = torch.distributions.Uniform(
                low=convert_frequencies_to_mels(
                    torch.tensor(
                        cutoff_freq_range[0],
                        dtype=torch.float32,
                        device=selected_samples.device,
                    )
                ),
                high=convert_frequencies_to_mels(
                    torch.tensor(
                        cutoff_freq_range[1],
                        dtype=torch.float32,
                        device=selected_samples.device,
                    )
                ),
                validate_args=True,
            )
            return dist

        low_dist = get_dist(self.low_cutoff_freq_range)
        high_dist = get_dist(self.high_cutoff_freq_range)

        self.transform_parameters["low_cutoff_freq"] = convert_mels_to_frequencies(
            low_dist.sample(sample_shape=(batch_size,))
        )
        self.transform_parameters["high_cutoff_freq"] = convert_mels_to_frequencies(
            high_dist.sample(sample_shape=(batch_size,))
        )

    def apply_transform(self, selected_samples: torch.Tensor, sample_rate: int = None):
        batch_size, num_channels, num_samples = selected_samples.shape

        if sample_rate is None:
            sample_rate = self.sample_rate

        low_cutoffs_as_fraction_of_sample_rate = (
            self.transform_parameters["low_cutoff_freq"] / sample_rate
        )
        high_cutoffs_as_fraction_of_sample_rate = (
            self.transform_parameters["high_cutoff_freq"] / sample_rate
        )
        # TODO: Instead of using a for loop, perform batched compute to speed things up
        for i in range(batch_size):
            selected_samples[i] = julius.bandpass_filter(
                selected_samples[i],
                cutoff_low=low_cutoffs_as_fraction_of_sample_rate[i].item(),
                cutoff_high=high_cutoffs_as_fraction_of_sample_rate[i].item(),
            )

        return selected_samples
