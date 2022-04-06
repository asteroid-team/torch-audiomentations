import torch
from typing import Optional
from torch import Tensor 

from ..core.transforms_interface import BaseWaveformTransform
from ..utils.object_dict import ObjectDict


class Padding(BaseWaveformTransform):

    supported_modes = {"per_batch", "per_example", "per_channel"}

    def __init__(
        self,
        min_fraction = 0.1,
        max_fraction = 0.5,
        pad_section = "end",
        mode = "per_batch",
        p = 0.5,
        p_model: Optional[str] = None,
        sample_rate: Optional[int] = None,
    ):
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
        self.pad_section = pad_section
        if not self.min_fraction>=0.0:
            raise ValueError("minimum fraction should be greater than zero.")
        if self.min_fraction<self.max_fraction:
            raise ValueError("minimum fraction should be greater than or equal to maximum fraction.")
        assert self.pad_section in ("start", "end"), 'pad_section must be "start" or "end"'



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


    def apply_tranform(self,
                       samples: Tensor,
                       sample_rate: Optional[int] = None,
                       targets:  Optional[int] = None,
                       target_rate: Optional[int] = None,
    
    ) -> ObjectDict:
        
        for i,index in enumerate(self.transform_parameters['pad_length']):
            if self.pad_section=="start":
                samples[i,:,:index] = 0.0
            else:
                samples[i,:,-index:] = 0.0
        
        return ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )




