from torch import Tensor
from typing import Optional

from ..augmentations.band_pass_filter import BandPassFilter
from ..utils.object_dict import ObjectDict


class BandStopFilter(BandPassFilter):
    """
    Apply band-stop filtering to the input audio. Also known as notch filter,
    band reject filter and frequency mask.
    """

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
        :param target_rate:
        """

        super().__init__(
            min_center_frequency,
            max_center_frequency,
            min_bandwidth_fraction,
            max_bandwidth_fraction,
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
        perturbed = super().apply_transform(
            samples.clone(),
            sample_rate,
            targets=targets.clone() if targets is not None else None,
            target_rate=target_rate,
        )

        perturbed.samples = samples - perturbed.samples
        return perturbed
