import torch
import torch.nn.functional as F

from ..core.transforms_interface import BaseWaveformTransform
from torch_pitch_shift import *
import math
from fractions import Fraction
from random import choices


class PitchShift(BaseWaveformTransform):
    """
    Pitch shift the sound up or down without changing the tempo
    """

    supports_multichannel = True
    requires_sample_rate = True

    def __init__(
        self,
        sample_rate: int,
        min_frequency_ratio=0.5,
        max_frequency_ratio=2,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
    ):
        """
        :param min_frequency_ratio: Minimum pitch shift ratio (default 0.5)
        :param max_frequency_ratio: Maximum pitch shift ratio (default 2)
        :param mode:
        :param p:
        :param p_mode:
        :param sample_rate:
        """
        super().__init__(mode, p, p_mode, sample_rate)

        self.min_frequency_ratio = Fraction(min_frequency_ratio)
        self.max_frequency_ratio = Fraction(max_frequency_ratio)
        if self.max_frequency_ratio > self.max_frequency_ratio:
            raise ValueError("max_shift_semitones must be > min_shift_semitones")
        self.pitch_shift = PitchShifter(
            sample_rate,
            lambda x: x >= self.min_frequency_ratio and x <= self.max_frequency_ratio,
        )

    def randomize_parameters(
        self, selected_samples: torch.Tensor, sample_rate: int = None
    ):
        """
        :params selected_samples: (batch_size, num_channels, num_samples)
        """
        batch_size, _, num_samples = selected_samples.shape

        self.transform_parameters["shift_ratio"] = choices(
            list(self.pitch_shift.fast_shifts), k=batch_size
        )

    def apply_transform(self, selected_samples: torch.Tensor, sample_rate: int = None):
        batch_size, num_channels, num_samples = selected_samples.shape

        if sample_rate is None:
            sample_rate = self.sample_rate

        for i in range(batch_size):
            selected_samples[i, ...] = self.pitch_shift(
                selected_samples[i],
                self.transform_parameters["shift_ratio"][i],
            )

        return selected_samples
