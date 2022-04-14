from turtle import forward
import torch
from typing import Optional
from torch import Tensor
import torchaudio


from ..core.transforms_interface import BaseWaveformTransform
from ..utils.dsp import convert_decibels_to_amplitude_ratio
from ..utils.object_dict import ObjectDict

class SpliceOut(BaseWaveformTransform):
    
    supported_modes = {"per_batch"}

    def __init__(
        self,
        num_time_intervals,
        max_width,
        mode: str = "per_batch",
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
        self.num_time_intervals = num_time_intervals
        self.max_width = max_width


    def randomize_parameters(
        self, samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None):

        self.transform_parameters["splice_length"] = torch.randint(low=0,
                                                    high=self.max_width,
                                                    size=(samples.shape[0],
                                                    self.num_time_intervals))




    def apply_transform(
        self, samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None) -> ObjectDict:

        for i in range(samples.shape[0]):
            spectrogram = torchaudio.transforms.Spectrogram()(samples[i])
            random_lengths = self.transform_parameters['splice_lengths'][i]
            mask = torch.ones(spectrogram.shape[-1],dtype=bool)
            for j in range(self.num_time_intervals):
                start = torch.randint(spectrogram.shape[-1]-random_lengths[j],size=(1,))
                mask[start:start+random_lengths[j]] = False




        
        return super().apply_transform(samples, sample_rate, targets, target_rate)


        
