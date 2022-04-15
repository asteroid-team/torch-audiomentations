from turtle import forward
import torch
from typing import Optional
from torch import Tensor

from ..core.transforms_interface import BaseWaveformTransform
from ..utils.dsp import convert_decibels_to_amplitude_ratio
from ..utils.object_dict import ObjectDict

class SpliceOut(BaseWaveformTransform):
    
    supported_modes = {"per_batch","per_example"}

    def __init__(
        self,
        num_time_intervals,
        max_width,
        mode: str = "per_example",
        n_fft = 400,
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
        self.n_fft = n_fft


    def randomize_parameters(
        self, samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None):

        self.transform_parameters["splice_lengths"] = torch.randint(low=0,
                                                    high=self.max_width,
                                                    size=(samples.shape[0],
                                                    self.num_time_intervals))

    def apply_transform(
        self, samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None) -> ObjectDict:

        spliceout_samples = []
        for i in range(samples.shape[0]):

            spectrogram = torch.stft(samples[i], n_fft=self.n_fft, return_complex=True)
            random_lengths = self.transform_parameters['splice_lengths'][i]
            mask = torch.ones(spectrogram.shape[-1], dtype=bool)
            
            for j in range(self.num_time_intervals):

                start = torch.randint(spectrogram.shape[-1]-random_lengths[j],size=(1,))
                mask[start:start+random_lengths[j]] = False
            
            spliceout_sample = torch.istft(spectrogram[:,:,mask], n_fft=self.n_fft)
            padding = torch.zeros((1,samples[i].shape[-1]-spliceout_sample.shape[-1]),dtype=torch.float32)
            spliceout_sample = torch.cat((spliceout_sample,padding),dim=-1)
            spliceout_samples.append(spliceout_sample)

        return torch.cat(spliceout_samples,dim=0)

        
