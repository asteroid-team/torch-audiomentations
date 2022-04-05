import torch
from typing import Optional
from torch import Tensor

from ..core.transforms_interface import BaseWaveformTransform

class Padding(BaseWaveformTransform):

    supported_modes = {"per_batch", "per_example"}

    def __init__(
        self,
        min_fraction = 0.1,
        max_fraction = 0.5,
        pad_mode = "silence",
        mode = "per_batch",
        p = 0.5,
        p_model: Optional[str] = None,
        sample_rate: Optional[int] = None,
    ):
        pass



    def randomize_parameters(self, samples: Tensor = None, sample_rate: Optional[int] = None, targets: Optional[Tensor] = None, target_rate: Optional[int] = None): 
        pass

    def apply_tranform(self):
        pass
