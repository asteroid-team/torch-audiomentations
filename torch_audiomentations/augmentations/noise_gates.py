import torch
from torch import Tensor
from typing import Optional

from ..core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.object_dict import ObjectDict



class SpectralGating(BaseWaveformTransform):

    def __init__(
        self,
        std_away = 1.0,
        n_grad_freq=2,
        n_grad_time=4,
        decrease_prop=1,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: Optional[str] = None,
        sample_rate: Optional[int] = None,
        target_rate: Optional[int] = None,
        output_type: Optional[str] = None,
    ):
        super(SpectralGating, self).__init__()


    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None):

        pass

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None) -> ObjectDict:

        pass
       