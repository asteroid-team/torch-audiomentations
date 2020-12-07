import torch
import typing

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

    requires_sample_rate = False

    def __init__(
        self,
        min_gain_in_db: float = -18.0,
        max_gain_in_db: float = 6.0,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: typing.Optional[str] = None,
        sample_rate: typing.Optional[int] = None,
    ):
        super().__init__(mode, p, p_mode, sample_rate)
        self.min_gain_in_db = min_gain_in_db
        self.max_gain_in_db = max_gain_in_db
        if self.min_gain_in_db >= self.max_gain_in_db:
            raise ValueError("max_gain_in_db must be higher than min_gain_in_db")

    def randomize_parameters(
        self, selected_samples, sample_rate: typing.Optional[int] = None
    ):
        distribution = torch.distributions.Uniform(
            low=torch.tensor(
                self.min_gain_in_db, dtype=torch.float32, device=selected_samples.device
            ),
            high=torch.tensor(
                self.max_gain_in_db, dtype=torch.float32, device=selected_samples.device
            ),
            validate_args=True,
        )
        selected_batch_size = selected_samples.size(0)
        self.transform_parameters["gain_factors"] = (
            convert_decibels_to_amplitude_ratio(
                distribution.sample(sample_shape=(selected_batch_size,))
            )
            .unsqueeze(1)
            .unsqueeze(1)
        )

    def apply_transform(self, selected_samples, sample_rate: typing.Optional[int] = None):
        return selected_samples * self.transform_parameters["gain_factors"]
