from ..core.transforms_interface import BaseWaveformTransform
from ..utils.object_dict import ObjectDict

from torch import Tensor
from typing import Optional
from random import choices
import torch


class Delay(BaseWaveformTransform):
    
    supported_modes = {"per_batch", "per_example"}

    supports_multichannel = True
    requires_sample_rate = False

    supports_target = True
    requires_target = False

    def __init__(
        self,
        min_delay_ms: float = 20.0,
        max_delay_ms: float = 100.0,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
        target_rate: int = None,
        output_type: Optional[str] = None,
        volume_factor: float = 0.5,
        repeats: int = 2,
        attenuation: float = 0.5
    ):
        """
        :param sample_rate:
        :param min_delay_ms: Minimum delay in milliseconds (default 20.0)
        :param max_delay_ms: Maximum delay in milliseconds (default 100.0)
        :param mode: ``per_example``, ``per_channel``, or ``per_batch``. Default ``per_example``.
        :param p:
        :param p_mode:
        :param target_rate:
        """
        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )

        if min_delay_ms > max_delay_ms:
            raise ValueError("max_delay_ms must be > min_delay_ms")
        if not sample_rate:
            raise ValueError("sample_rate is invalid.")
        self._sample_rate = sample_rate
        self._max_delay_ms = max_delay_ms
        self._min_delay_ms = min_delay_ms
        self._max_delay_samples = int(max_delay_ms * sample_rate / 1000)
        self._min_delay_samples = int(min_delay_ms * sample_rate / 1000)
        self._mode = mode
        self._volume_factor = volume_factor
        self._repeats = repeats
        self._attenuation = attenuation

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        """
        :param samples: (batch_size, num_channels, num_samples)
        :param sample_rate:
        """
        batch_size, num_channels, num_samples = samples.shape

        if self._mode == "per_example":
            self.transform_parameters["delays"] = choices(
                range(self._min_delay_samples, self._max_delay_samples + 1), k=batch_size
            )
            self.transform_parameters['volume_factors'] = choices([self._volume_factor-0.2, self._volume_factor+0.2],k=batch_size)
            self.transform_parameters['repeats'] = choices([self._repeats-1, self._repeats+1],k=batch_size)
            self.transform_parameters['attenuation'] = choices([self._attenuation-0.2, self._attenuation+0.2],k=batch_size)
        
        elif self._mode == "per_batch":
            self.transform_parameters["delays"] = choices(
                range(self._min_delay_samples, self._max_delay_samples + 1), k=1
            )
            self.transform_parameters['volume_factors'] = choices([self._volume_factor-0.2, self._volume_factor+0.2],k=1)
            self.transform_parameters['repeats'] = choices([self._repeats-1, self._repeats+1],k=1)
            self.transform_parameters['attenuation'] = choices([self._attenuation-0.2, self._attenuation+0.2],k=1)
            
    def apply_transform(
        
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        """
        :param samples: (batch_size, num_channels, num_samples)
        :param sample_rate:
        """
        batch_size, num_channels, num_samples = samples.shape


        if self._mode == "per_example":
            for i in range(batch_size):
                samples[i, ...] = self.delay(
                    samples[i][None],
                    self.transform_parameters["delays"][i],
                    self.transform_parameters['volume_factors'][i],
                    self.transform_parameters['repeats'][i],
                    self.transform_parameters['attenuation'][i],
                )[0]


        elif self._mode == "per_batch":
            samples = self.delay(
                samples, self.transform_parameters["delays"][0], self.transform_parameters['volume_factors'][0], 
                self.transform_parameters['repeats'][0],
                self.transform_parameters['attenuation'][0],
            )

        return ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )
        
    def delay(self,samples: Tensor, delay_samples: int, volume_factor: float , repeats, attenuation) -> Tensor:
        
        ## add the original signal delayed by delay_samples to the original signal
        ## do this operation repeats times and each time attenuate the delayed signal by attenuation factor
        ## reduce the whole delayed signal by volume_factor
        ## then add both signals together and truncate to be the same length as the original signal
        
        batch_size, num_channels, num_samples = samples.shape
        
        delayed_signal = torch.zeros(batch_size, num_channels, num_samples + repeats * delay_samples, device = samples.device)
        
        for i in range(repeats):
            delayed_signal[:,:,i*delay_samples:i*delay_samples+num_samples] += samples * attenuation ** (i)
            
        delayed_signal = delayed_signal[:,:,:num_samples]
        
        samples = (samples + volume_factor * delayed_signal)/2

        return samples