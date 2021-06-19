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
        min_transpose=-12,
        max_transpose=12,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
    ):
        """
        :param min_transpose: Minimum pitch shift transposition (default -12 semitones)
        :param max_transpose: Maximum pitch shift transposition (default +12 semitones)
        :param mode:
        :param p:
        :param p_mode:
        :param sample_rate:
        """
        super().__init__(mode, p, p_mode, sample_rate)

        self.min_transpose = min_transpose
        self.max_transpose = max_transpose
        if self.min_transpose > self.max_transpose:
            raise ValueError("max_transpose must be > min_transpose")
        self.pitch_shift = PitchShifter()

    def randomize_parameters(
        self, selected_samples: torch.Tensor, sample_rate: int = None
    ):
        """
        :params selected_samples: (batch_size, num_channels, num_samples)
        """
        batch_size, _, num_samples = selected_samples.shape

        dist = torch.distributions.Uniform(
            low=self.min_transpose,
            high=self.max_transpose,
            validate_args=True,
        )
        self.transform_parameters["transpositions"] = dist.sample(
            sample_shape=(batch_size,)
        )

    def apply_transform(self, selected_samples: torch.Tensor, sample_rate: int = None):
        batch_size, num_channels, num_samples = selected_samples.shape

        if sample_rate is None:
            sample_rate = self.sample_rate

        for i in range(batch_size):
            selected_samples[i, ...] = self.pitch_shift(
                selected_samples[i],
                self.transform_parameters["transpositions"][i],
                sample_rate,
            )

        return selected_samples
