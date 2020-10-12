import torch

from ..core.transforms_interface import BaseWaveformTransform
from ..utils.dsp import convert_decibels_to_amplitude_ratio


class Gain(BaseWaveformTransform):
    """
    Multiply the audio by a random amplitude factor to reduce or increase the volume. This
    technique can help a model become somewhat invariant to the overall gain of the input audio.

    Warning: This transform can return samples outside the [-1, 1] range, which may lead to
    clipping or wrap distortion, depending on what you do with the audio in a later stage.
    See also https://en.wikipedia.org/wiki/Clipping_(audio)#Digital_clipping
    """

    supports_multichannel = True

    def __init__(
        self, min_gain_in_db: float = -18.0, max_gain_in_db: float = 6.0, p: float = 0.5
    ):
        """
        :param p:
        """
        super().__init__(p)
        self.register_buffer(
            name="_min_gain_in_db",
            tensor=torch.tensor(min_gain_in_db, dtype=torch.float32),
        )
        self.register_buffer(
            name="_max_gain_in_db",
            tensor=torch.tensor(max_gain_in_db, dtype=torch.float32),
        )
        self.gain_distribution = self.reset_distribution()

    def reset_distribution(self):
        self.gain_distribution = torch.distributions.Uniform(
            low=self._min_gain_in_db, high=self._max_gain_in_db, validate_args=True
        )
        return self.gain_distribution

    def to(self, *args, **kwargs):
        return_value = super().to(*args, **kwargs)
        self.reset_distribution()
        return return_value

    def cuda(self, *args, **kwargs):
        return_value = super().cuda(*args, **kwargs)
        self.reset_distribution()
        return return_value

    def cpu(self, *args, **kwargs):
        return_value = super().cpu(*args, **kwargs)
        self.reset_distribution()
        return return_value

    @property
    def min_gain_in_db(self):
        return self._min_gain_in_db

    @min_gain_in_db.setter
    def min_gain_in_db(self, min_gain_in_db):
        self._min_gain_in_db = torch.tensor(
            min_gain_in_db, dtype=torch.float32, device=self._min_gain_in_db.device
        )
        self.reset_distribution()

    @property
    def max_gain_in_db(self):
        return self._max_gain_in_db

    @max_gain_in_db.setter
    def max_gain_in_db(self, max_gain_in_db):
        self._max_gain_in_db = torch.tensor(
            max_gain_in_db, dtype=torch.float32, device=self._max_gain_in_db.device
        )
        self.reset_distribution()

    def randomize_parameters(self, selected_samples, sample_rate: int):
        selected_batch_size = selected_samples.size(0)
        self.parameters["gain_factors"] = convert_decibels_to_amplitude_ratio(
            self.gain_distribution.sample(sample_shape=(selected_batch_size,))
        )

    def apply_transform(self, selected_samples, sample_rate: int):
        num_dimensions = len(selected_samples.shape)
        if num_dimensions == 1:
            gain_factors = self.parameters["gain_factors"]
        elif num_dimensions == 2:
            gain_factors = self.parameters["gain_factors"].unsqueeze(1)
        elif num_dimensions == 3:
            gain_factors = self.parameters["gain_factors"].unsqueeze(1).unsqueeze(1)
        else:
            raise Exception(
                "Invalid number of dimensions ({}) in input tensor".format(num_dimensions)
            )
        return selected_samples * gain_factors
