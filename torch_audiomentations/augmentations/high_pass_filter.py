from torch import Tensor
from typing import Optional

from ..augmentations.low_pass_filter import LowPassFilter
from ..utils.object_dict import ObjectDict


class HighPassFilter(LowPassFilter):
    """
    Apply high-pass filtering to the input audio.
    """

    def __init__(
        self,
        min_cutoff_freq: float = 20.0,
        max_cutoff_freq: float = 2400.0,
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
        :param target_rate:
        """

        super().__init__(
            min_cutoff_freq,
            max_cutoff_freq,
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
            samples=samples.clone(),
            sample_rate=sample_rate,
            targets=targets.clone() if targets is not None else None,
            target_rate=target_rate,
        )

        perturbed.samples = samples - perturbed.samples
        return perturbed
