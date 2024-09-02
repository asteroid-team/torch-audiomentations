import torch
from torch import Tensor
from typing import Optional

from ..core.transforms_interface import BaseWaveformTransform
from ..utils.dsp import convert_decibels_to_amplitude_ratio
from ..utils.object_dict import ObjectDict


class Gain(BaseWaveformTransform):
    """
    Multiply the audio by a random amplitude factor to reduce or increase the volume. This
    technique can help a model become somewhat invariant to the overall gain of the input audio.

    Warning: This transform can return samples outside the [-1, 1] range, which may lead to
    clipping or wrap distortion, depending on what you do with the audio in a later stage.
    See also https://en.wikipedia.org/wiki/Clipping_(audio)#Digital_clipping
    """

    supported_modes = {"per_batch", "per_example", "per_channel"}

    supports_multichannel = True
    requires_sample_rate = False

    supports_target = True
    requires_target = False

    def __init__(
        self,
        min_gain_in_db: float = -18.0,
        max_gain_in_db: float = 6.0,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: Optional[str] = None,
        sample_rate: Optional[int] = None,
        target_rate: Optional[int] = None,
        output_type: Optional[str] = None,
    ):
        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )
        self.min_gain_in_db = min_gain_in_db
        self.max_gain_in_db = max_gain_in_db
        if self.min_gain_in_db >= self.max_gain_in_db:
            raise ValueError("max_gain_in_db must be higher than min_gain_in_db")

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        distribution = torch.distributions.Uniform(
            low=torch.tensor(
                self.min_gain_in_db, dtype=torch.float32, device=samples.device
            ),
            high=torch.tensor(
                self.max_gain_in_db, dtype=torch.float32, device=samples.device
            ),
            validate_args=True,
        )
        selected_batch_size = samples.size(0)
        self.transform_parameters["gain_factors"] = (
            convert_decibels_to_amplitude_ratio(
                distribution.sample(sample_shape=(selected_batch_size,))
            )
            .unsqueeze(1)
            .unsqueeze(1)
        )

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        return ObjectDict(
            samples=samples * self.transform_parameters["gain_factors"],
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )
