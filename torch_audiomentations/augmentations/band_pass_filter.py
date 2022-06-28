import julius
import torch
from torch import Tensor
from typing import Optional
from ..core.transforms_interface import BaseWaveformTransform
from ..utils.mel_scale import convert_frequencies_to_mels, convert_mels_to_frequencies
from ..utils.object_dict import ObjectDict


class BandPassFilter(BaseWaveformTransform):
    """
    Apply band-pass filtering to the input audio.
    """

    supported_modes = {"per_batch", "per_example", "per_channel"}

    supports_multichannel = True
    requires_sample_rate = True

    supports_target = True
    requires_target = False

    def __init__(
        self,
        min_center_frequency=200,
        max_center_frequency=4000,
        min_bandwidth_fraction=0.5,
        max_bandwidth_fraction=1.99,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
        target_rate: int = None,
        output_type: Optional[str] = None,
    ):
        """
        :param min_center_frequency: Minimum center frequency in hertz
        :param max_center_frequency: Maximum center frequency in hertz
        :param min_bandwidth_fraction: Minimum bandwidth fraction relative to center
            frequency (number between 0.0 and 2.0)
        :param max_bandwidth_fraction: Maximum bandwidth fraction relative to center
            frequency (number between 0.0 and 2.0)
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

        self.min_center_frequency = min_center_frequency
        self.max_center_frequency = max_center_frequency
        self.min_bandwidth_fraction = min_bandwidth_fraction
        self.max_bandwidth_fraction = max_bandwidth_fraction

        if max_center_frequency < min_center_frequency:
            raise ValueError(
                f"max_center_frequency ({max_center_frequency}) should be larger than "
                f"min_center_frequency ({min_center_frequency})."
            )

        if min_bandwidth_fraction <= 0.0:
            raise ValueError("min_bandwidth_fraction must be a positive number")

        if max_bandwidth_fraction < min_bandwidth_fraction:
            raise ValueError(
                f"max_bandwidth_fraction ({max_bandwidth_fraction}) should be larger than "
                f"min_bandwidth_fraction ({min_bandwidth_fraction})."
            )

        if max_bandwidth_fraction >= 2.0:
            raise ValueError(
                f"max_bandwidth_fraction ({max_bandwidth_fraction}) should be smaller than 2.0,"
                f"since otherwise low_cut_frequency of the band can be smaller than 0 Hz."
            )

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        """
        :params samples: (batch_size, num_channels, num_samples)
        """

        batch_size, _, num_samples = samples.shape

        # Sample frequencies uniformly in mel space, then convert back to frequency
        def get_dist(min_freq, max_freq):
            dist = torch.distributions.Uniform(
                low=convert_frequencies_to_mels(
                    torch.tensor(min_freq, dtype=torch.float32, device=samples.device)
                ),
                high=convert_frequencies_to_mels(
                    torch.tensor(max_freq, dtype=torch.float32, device=samples.device)
                ),
                validate_args=True,
            )
            return dist

        center_dist = get_dist(self.min_center_frequency, self.max_center_frequency)
        self.transform_parameters["center_freq"] = convert_mels_to_frequencies(
            center_dist.sample(sample_shape=(batch_size,))
        )

        bandwidth_dist = torch.distributions.Uniform(
            low=torch.tensor(
                self.min_bandwidth_fraction, dtype=torch.float32, device=samples.device
            ),
            high=torch.tensor(
                self.max_bandwidth_fraction, dtype=torch.float32, device=samples.device
            ),
        )
        self.transform_parameters["bandwidth"] = bandwidth_dist.sample(
            sample_shape=(batch_size,)
        )

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        batch_size, num_channels, num_samples = samples.shape

        low_cutoffs_as_fraction_of_sample_rate = (
            self.transform_parameters["center_freq"]
            * (1 - 0.5 * self.transform_parameters["bandwidth"])
            / sample_rate
        )
        high_cutoffs_as_fraction_of_sample_rate = (
            self.transform_parameters["center_freq"]
            * (1 + 0.5 * self.transform_parameters["bandwidth"])
            / sample_rate
        )
        # TODO: Instead of using a for loop, perform batched compute to speed things up
        for i in range(batch_size):
            samples[i] = julius.bandpass_filter(
                samples[i],
                cutoff_low=low_cutoffs_as_fraction_of_sample_rate[i].item(),
                cutoff_high=high_cutoffs_as_fraction_of_sample_rate[i].item(),
            )

        return ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )
