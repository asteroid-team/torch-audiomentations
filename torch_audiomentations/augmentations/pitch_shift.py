from random import choices
import torch
import torch.nn.functional as F

from ..core.transforms_interface import BaseWaveformTransform
from torch_pitch_shift import *


class PitchShift(BaseWaveformTransform):
    """
    Pitch-shift the sound up or down without changing the tempo
    """

    supports_multichannel = True
    requires_sample_rate = True

    def __init__(
        self,
        sample_rate: int,
        min_transpose_ratio=0.5,
        max_transpose_ratio=2,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
    ):
        """
        :param min_transpose_ratio: Minimum pitch shift transposition ratio (default 0.5 --> -1 octaves)
        :param max_transpose_ratio: Maximum pitch shift transposition ratio (default 2 --> +1 octaves)
        :param mode:
        :param p:
        :param p_mode:
        :param sample_rate:
        """
        super().__init__(mode, p, p_mode, sample_rate)

        self._min_transpose_ratio = min_transpose_ratio
        self._max_transpose_ratio = max_transpose_ratio
        if self._min_transpose_ratio > self._max_transpose_ratio:
            raise ValueError("max_transpose_ratio must be > min_transpose_ratio")
        self._pitch_shift = PitchShifter()
        self._sample_rate = sample_rate
        self._fast_shifts = self.fast_shifts = get_fast_shifts(
            sample_rate, lambda x: x >= min_transpose_ratio and x <= max_transpose_ratio
        )

    def randomize_parameters(
        self, selected_samples: torch.Tensor, sample_rate: int = None
    ):
        """
        :params selected_samples: (batch_size, num_channels, num_samples)
        """
        batch_size, _, num_samples = selected_samples.shape

        self.transform_parameters["transpositions"] = choices(
            self._fast_shifts, k=batch_size
        )

    def apply_transform(self, selected_samples: torch.Tensor, sample_rate: int = None):
        batch_size, num_channels, num_samples = selected_samples.shape

        if sample_rate is None:
            sample_rate = self._sample_rate

        for i in range(batch_size):
            selected_samples[i, ...] = self._pitch_shift(
                selected_samples[i],
                self.transform_parameters["transpositions"][i],
                sample_rate,
            )

        return selected_samples
