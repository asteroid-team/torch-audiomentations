from random import choices

import torch
from torch_pitch_shift import pitch_shift, get_fast_shifts, semitones_to_ratio

from ..core.transforms_interface import BaseWaveformTransform


class PitchShift(BaseWaveformTransform):
    """
    Pitch-shift sounds up or down without changing the tempo.
    """

    supports_multichannel = True
    requires_sample_rate = True

    def __init__(
        self,
        sample_rate: int,
        min_transpose_semitones: float = -4.0,
        max_transpose_semitones: float = 4.0,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
    ):
        """
        :param sample_rate:
        :param min_transpose_semitones: Minimum pitch shift transposition in semitones (default -4.0)
        :param max_transpose_semitones: Maximum pitch shift transposition in semitones (default +4.0)
        :param mode: ``per_example``, ``per_channel``, or ``per_batch``. Default ``per_example``.
        :param p:
        :param p_mode:
        """
        super().__init__(mode, p, p_mode, sample_rate)

        if min_transpose_semitones > max_transpose_semitones:
            raise ValueError("max_transpose_semitones must be > min_transpose_semitones")
        if not sample_rate:
            raise ValueError("sample_rate is invalid.")
        self._sample_rate = sample_rate
        self._fast_shifts = get_fast_shifts(
            sample_rate,
            lambda x: x >= semitones_to_ratio(min_transpose_semitones)
            and x <= semitones_to_ratio(max_transpose_semitones)
            and x != 1,
        )
        if not len(self._fast_shifts):
            raise ValueError(
                "No fast pitch-shift ratios could be computed for the given sample rate and transpose range."
            )
        self._mode = mode

    def randomize_parameters(
        self, selected_samples: torch.Tensor, sample_rate: int = None
    ):
        """
        :param selected_samples: (batch_size, num_channels, num_samples)
        :param sample_rate:
        """
        batch_size, num_channels, num_samples = selected_samples.shape

        if self._mode == "per_example":
            self.transform_parameters["transpositions"] = choices(
                self._fast_shifts, k=batch_size
            )
        elif self._mode == "per_channel":
            self.transform_parameters["transpositions"] = list(
                zip(
                    *[
                        choices(self._fast_shifts, k=batch_size)
                        for i in range(num_channels)
                    ]
                )
            )
        elif self._mode == "per_batch":
            self.transform_parameters["transpositions"] = choices(self._fast_shifts, k=1)

    def apply_transform(self, selected_samples: torch.Tensor, sample_rate: int = None):
        """
        :param selected_samples: (batch_size, num_channels, num_samples)
        :param sample_rate:
        """
        batch_size, num_channels, num_samples = selected_samples.shape

        if sample_rate is not None and sample_rate != self._sample_rate:
            raise ValueError(
                "sample_rate must match the value of sample_rate "
                + "passed into the PitchShift constructor"
            )
        sample_rate = self.sample_rate

        if self._mode == "per_example":
            for i in range(batch_size):
                selected_samples[i, ...] = pitch_shift(
                    selected_samples[i][None],
                    self.transform_parameters["transpositions"][i],
                    sample_rate,
                )[0]
        elif self._mode == "per_channel":
            for i in range(batch_size):
                for j in range(num_channels):
                    selected_samples[i, j, ...] = pitch_shift(
                        selected_samples[i][j][None][None],
                        self.transform_parameters["transpositions"][i][j],
                        sample_rate,
                    )[0][0]
        elif self._mode == "per_batch":
            return pitch_shift(
                selected_samples,
                self.transform_parameters["transpositions"][0],
                sample_rate,
            )

        return selected_samples
