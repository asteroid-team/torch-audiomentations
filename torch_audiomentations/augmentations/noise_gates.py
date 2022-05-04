from dis import dis
import torch
from torch import Tensor
from typing import Optional
from torchaudio.transforms import AmplitudeToDB

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
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.decrease_prop = min(1.0,decrease_prop)  ##max decrease prop



    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None):

        dist = torch.distributions.Uniform(0.0,self.decrease_prop)
        self.transform_parameters['decrease_prop'] = dist.sample((samples.shape[0],))

        

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None) -> ObjectDict:

        ### threshold 
        if self.mode == "per_batch":
            assert samples.shape[0]==1
            audio_stft_abs = torch.stft(samples.squeeze(0),n_fft=self.n_fft,win_length=self.win_length,hop_length=self.hop_length)[:,:,:,0]
            audio_stft_db = AmplitudeToDB()(audio_stft_abs)
            audio_mean_db = audio_stft_db.mean(dim=0)
            noise_threshold = torch.quantile(audio_mean_db,q=self.q,dim=-1)
            noise_threshold = noise_threshold.unsqueeze(-1).expand(audio_mean_db.shape)
        elif self.mode == "per_example":
            audio_stft_abs = torch.stft(samples.mean(dim=1),n_fft=self.n_fft,win_length=self.win_length,hop_length=self.hop_length)[:,:,:,0]
            audio_stft_db = AmplitudeToDB()(audio_stft_abs)
            noise_threshold = torch.quantile(audio_stft_db,q=self.q,dim=-1)
            noise_threshold = noise_threshold.unsqueeze(-1).expand(audio_stft_db.shape)
        else:
            audio_stft_abs = torch.stft(samples.squeeze(1),n_fft=self.n_fft,win_length=self.win_length,hop_length=self.hop_length)[:,:,:,0]
            audio_stft_db = AmplitudeToDB()(audio_stft_abs)
            noise_threshold = torch.quantile(audio_stft_db,q=self.q,dim=-1)
            noise_threshold = noise_threshold.unsqueeze(-1).expand(audio_stft_db.shape)


        





                                   ### (1,batch_size*num_channels, num_samples)
                            ### (batch_size,num_channels,num_samples)  
                            ### (batch_size*num_channels,1,num_samples)
    
       