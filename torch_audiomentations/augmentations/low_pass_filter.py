import julius
import torch
from torch import Tensor
from typing import Optional


from ..core.transforms_interface import BaseWaveformTransform
from ..utils.mel_scale import convert_frequencies_to_mels, convert_mels_to_frequencies
from ..utils.object_dict import ObjectDict


class LowPassFilter(BaseWaveformTransform):
    """
    Apply low-pass filtering to the input audio.
    """

    supported_modes = {"per_batch", "per_example", "per_channel"}

    supports_multichannel = True
    requires_sample_rate = True

    supports_target = True
    requires_target = False

    def __init__(
        self,
        min_cutoff_freq=150,
        max_cutoff_freq=7500,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
        target_rate: int = None,
        output_type: Optional[str] = None,
    ):
        """
        :param min_cutoff_freq: Minimum cutoff frequency in hertz
        :param max_cutoff_freq: Maximum cutoff frequency in hertz
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

        self.min_cutoff_freq = min_cutoff_freq
        self.max_cutoff_freq = max_cutoff_freq
        if self.min_cutoff_freq > self.max_cutoff_freq:
            raise ValueError("min_cutoff_freq must not be greater than max_cutoff_freq")

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
        dist = torch.distributions.Uniform(
            low=convert_frequencies_to_mels(
                torch.tensor(
                    self.min_cutoff_freq, dtype=torch.float32, device=samples.device
                )
            ),
            high=convert_frequencies_to_mels(
                torch.tensor(
                    self.max_cutoff_freq, dtype=torch.float32, device=samples.device
                )
            ),
            validate_args=True,
        )
        self.transform_parameters["cutoff_freq"] = convert_mels_to_frequencies(
            dist.sample(sample_shape=(batch_size,))
        )

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:

        batch_size, num_channels, num_samples = samples.shape

        cutoffs_as_fraction_of_sample_rate = (
            self.transform_parameters["cutoff_freq"] / sample_rate
        )
        # TODO: Instead of using a for loop, perform batched compute to speed things up
        for i in range(batch_size):
            samples[i] = julius.lowpass_filter(
                samples[i], cutoffs_as_fraction_of_sample_rate[i].item()
            )

        return ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )
