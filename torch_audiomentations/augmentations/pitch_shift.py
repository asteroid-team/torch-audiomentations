import torch

from ..core.transforms_interface import BaseWaveformTransform


class LowPassFilter(BaseWaveformTransform):
    """
    Pitch shift the sound up or down without changing the tempo
    """

    supports_multichannel = True
    requires_sample_rate = True

    def __init__(
        self,
        min_shift_semitones=-4,
        max_shift_semitones=4,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
    ):
        """
        :param min_shift_semitones: Minimum pitch shift in semitones
        :param max_shift_semitones: Maximum pitch shift in semitones
        :param mode:
        :param p:
        :param p_mode:
        :param sample_rate:
        """
        super().__init__(mode, p, p_mode, sample_rate)

        self.min_shift_semitones = min_shift_semitones
        self.max_shift_semitones = max_shift_semitones
        if max(abs(min_shift_semitones), abs(max_shift_semitones)) >= 12:
            raise ValueError(
                "Magnitude of max_shift_semitones and min_shift_semitones must be < 12"
            )
        if self.min_shift_semitones > self.max_shift_semitones:
            raise ValueError("max_shift_semitones must be > min_shift_semitones")

    def randomize_parameters(
        self, selected_samples: torch.Tensor, sample_rate: int = None
    ):
        """
        :params selected_samples: (batch_size, num_channels, num_samples)
        """
        batch_size, _, num_samples = selected_samples.shape

        # Sample frequencies uniformly in mel space, then convert back to frequency
        dist = torch.distributions.Uniform(
            low=self.min_shift_semitones,
            high=self.max_shift_semitones,
            validate_args=True,
        )
        self.transform_parameters["num_semitones"] = dist.sample(
            sample_shape=(batch_size,)
        )

    def apply_transform(self, selected_samples: torch.Tensor, sample_rate: int = None):
        batch_size, num_channels, num_samples = selected_samples.shape

        if sample_rate is None:
            sample_rate = self.sample_rate

        # TODO: import my own library "torch_pitch_shift"
        return tps.pitch_shift(selected_samples, n_steps=self.parameters["num_semitones"])
