import torch
import numpy as np
from torch import Tensor
from typing import Optional
from torchaudio.functional import DB_to_amplitude,amplitude_to_DB

from ..core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.object_dict import ObjectDict



class SpectralGating(BaseWaveformTransform):

    supported_modes = {"per_batch", "per_example", "per_channel"}

    def __init__(
        self,
        std_away = 1.0,
        n_grad_freq=2,
        n_grad_time=4,
        q = 0.1,
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
        super(SpectralGating, self).__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.decrease_prop = min(1.0,decrease_prop)  ##max decrease prop
        self.q = q
        self.n_grad_freq = n_grad_freq
        self.n_grad_time = n_grad_time



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


                            ### (1,batch_size*num_channels, num_samples)
                            ### (batch_size,num_channels,num_samples)  
                            ### (batch_size*num_channels,1,num_samples)

        ### threshold 
        if self.mode == "per_batch":
            assert samples.shape[0]==1
            audio_stft_abs = torch.stft(samples.squeeze(0),n_fft=self.n_fft,win_length=self.win_length,hop_length=self.hop_length)[:,:,:,0]
            audio_stft_db = amplitude_to_DB(audio_stft_abs, multiplier=20, amin=1e-10, db_multiplier=0.0)
            audio_mean_db = audio_stft_db.mean(dim=0)
            noise_threshold = torch.quantile(audio_mean_db,q=self.q,dim=-1)
            noise_threshold = noise_threshold.unsqueeze(-1).expand(audio_mean_db.shape)[(None,)*2].expand((samples.shape[0],samples.shape[1],audio_stft_abs.shape[1],audio_stft_abs.shape[2]))
        elif self.mode == "per_example":
            audio_stft_abs = torch.stft(samples.mean(dim=1),n_fft=self.n_fft,win_length=self.win_length,hop_length=self.hop_length)[:,:,:,0]
            audio_stft_db = amplitude_to_DB(audio_stft_abs, multiplier=20, amin=1e-10, db_multiplier=0.0)
            noise_threshold = torch.quantile(audio_stft_db,q=self.q,dim=-1)
            noise_threshold = noise_threshold.unsqueeze(-1).expand(audio_stft_db.shape).unsqueeze(1).expand((samples.shape[0],samples.shape[1],audio_stft_abs.shape[1],audio_stft_abs.shape[2]))
        else:
            audio_stft_abs = torch.stft(samples.squeeze(1),n_fft=self.n_fft,win_length=self.win_length,hop_length=self.hop_length)[:,:,:,0]
            audio_stft_db = amplitude_to_DB(audio_stft_abs, multiplier=20, amin=1e-10, db_multiplier=0.0)
            noise_threshold = torch.quantile(audio_stft_db,q=self.q,dim=-1)
            noise_threshold = noise_threshold.unsqueeze(-1).expand(audio_stft_db.shape).unsqueeze(1)

        print("SAMPLES",samples.shape,"THRESHOLD",noise_threshold.shape,"MODE",self.mode)
        smoothing_filter = torch.outer(
            torch.cat( 
                (torch.linspace(0,1,self.n_grad_freq+1),
                torch.linspace(1,0,self.n_grad_freq+2))
            )[1:-1],
            torch.cat(
                (torch.linspace(0,1,self.n_grad_time+1),
                torch.linspace(1,0,self.n_grad_time+2)
                )

            )[1:-1]
        )
        smoothing_filter = smoothing_filter/smoothing_filter.sum()
        print("Smoothing", smoothing_filter.shape)
        cleaned_audios = []
        for i,sample,noise_thresh_matrix in zip(np.arange(0,samples.shape[0]),samples,noise_threshold):
            for sample_dim,noise_dim in zip(sample,noise_thresh_matrix):
                audio_stft = torch.stft(sample_dim,n_fft=self.n_fft,win_length=self.win_length,hop_length=self.hop_length)
                audio_stft_db = amplitude_to_DB(audio_stft[:,:,0], multiplier=20, amin=1e-10, db_multiplier=0.0)
                mask_gain_db = torch.min(audio_stft_db)
                noise_mask = audio_stft_db < noise_dim
                with torch.no_grad():
                    noise_mask = torch.nn.functional.conv2d(noise_mask.float()[(None,)*2],smoothing_filter[(None,)*2],padding="same")[0,0,:,:] * self.transform_parameters['decrease_prop'][i]
                cleaned_audio_real = audio_stft_db * (1-noise_mask) + torch.ones(mask_gain_db.shape) * mask_gain_db * noise_mask
                cleaned_audio_img = audio_stft[:,:,1] * (1-noise_mask)

                cleaned_audio_stft = torch.stack(
                                    (DB_to_amplitude(cleaned_audio_real,ref=1,power=0.5)*audio_stft[:,:,0].sign(),
                                    cleaned_audio_img),dim=-1)
                cleaned_audio = torch.istft(cleaned_audio_stft,hop_length=self.hop_length,win_length=self.win_length,n_fft=self.n_fft).unsqueeze(0)
                padding = torch.zeros((1,sample.shape[-1]-cleaned_audio.shape[1]),device=sample.device)
                print("PADDING",padding.shape,"CLEANED",cleaned_audio.shape)
                cleaned_audio = torch.cat((cleaned_audio,padding),dim=1)
                cleaned_audios.append(cleaned_audio)
    
        return ObjectDict(
            samples=torch.cat(cleaned_audios,dim=0).reshape(samples.shape),
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )
       