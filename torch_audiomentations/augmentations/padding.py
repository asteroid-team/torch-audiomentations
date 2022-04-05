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
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
        self.pad_mode = pad_mode
        if not self.min_fraction>=0.0:
            raise ValueError("minimum fraction should be greater than zero.")
        if self.min_fraction<self.max_fraction:
            raise ValueError("minimum fraction should be greater than or equal to maximum fraction.")
        assert self.pad_mode in ("silence", "wrap", "reflect"), 'pad_mode must be "silence", "wrap" or "reflect"'



    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None
    ):  
        input_length = samples.shape[-1]
        self.transform_parameters["pad_length"] = torch.randint(
                                                    int(input_length*self.min_fraction),
                                                    int(input_length*self.max_fraction),
                                                    (samples.shape[0],)
        )

        

    def apply_tranform(self):
        pass
