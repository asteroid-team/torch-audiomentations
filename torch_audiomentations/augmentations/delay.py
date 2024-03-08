from ..core.transforms_interface import BaseWaveformTransform
from ..utils.object_dict import ObjectDict

from torch import Tensor
from typing import Optional
from random import choices
import torch
import numpy as np


class Delay(BaseWaveformTransform):
    """ 
    A rudimentary delay effect which delays the original signal multiple times and applies an attenuation factor to each delayed signal.
    The delayed signals are then added to the original signal and the whole delayed signal is reduced by a volume factor.
    """
    
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
        volume_factor_range: float = 0.2,
        repeats: int = 2,
        repeats_range: int = 1,
        attenuation: float = 0.5,
        attenuation_range:int = 0.2
    ):
        """
        :param sample_rate:
        :param min_delay_ms: Minimum delay in milliseconds (default 20.0)
        :param max_delay_ms: Maximum delay in milliseconds (default 100.0)
        :param mode: ``per_example``, ``per_channel``, or ``per_batch``. Default ``per_example``.
        :param p:
        :param p_mode:
        :param target_rate:
        :param output_type:
        :param volume_factor: The factor by which the delayed signal is reduced compared to the original signal. Default 0.5
        :param volume_factor_range: The range of the volume factor. Default 0.2
        :param repeats: The number of times the delayed signal is added to the original signal. Default 2
        :param repeats_range: The range of the number of repeats. Default 1
        :param attenuation: The factor by which the delayed signal is attenuated for each repeat. Default 0.5
        :param attenuation_range: The range of the attenuation factor. Default 0.2
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
        self._volume_factor_range = volume_factor_range
        self._repeats = repeats
        self._repeats_range = repeats_range
        self._attenuation = attenuation
        self._attenuation_range = attenuation_range

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
            self.transform_parameters['volume_factors'] = np.random.uniform(self._volume_factor-self._volume_factor_range, self._volume_factor+self._volume_factor_range, batch_size)
            self.transform_parameters['repeats'] = choices([self._repeats-self._repeats_range, self._repeats+self._repeats_range],k=batch_size)
            self.transform_parameters['attenuation'] = choices([self._attenuation-self._attenuation_range, self._attenuation+self._attenuation_range],k=batch_size)
        
        elif self._mode == "per_batch":
            self.transform_parameters["delays"] = choices(
                range(self._min_delay_samples, self._max_delay_samples + 1), k=1
            )
            self.transform_parameters['volume_factors'] = np.random.uniform(self._volume_factor-self._volume_factor_range, self._volume_factor+self._volume_factor_range, 1)
            self.transform_parameters['repeats'] = choices([self._repeats-self._repeats_range, self._repeats+self._repeats_range],k=1)
            self.transform_parameters['attenuation'] = choices([self._attenuation-self._attenuation_range, self._attenuation+self._attenuation_range],k=1)
            
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
        
    def delay(self,samples: Tensor, delay_samples: int, volume_factor: float , repeats : int, attenuation : float) -> Tensor:
        """ 
        :param samples: (batch_size, num_channels, num_samples)
        :param delay_samples: int
        :param volume_factor: float
        :param repeats: int
        :param attenuation: float
        
        """

        batch_size, num_channels, num_samples = samples.shape
        
        delayed_signal = torch.zeros(batch_size, num_channels, num_samples + repeats * delay_samples, device = samples.device)
        
        for i in range(repeats):
            delayed_signal[:,:,i*delay_samples:i*delay_samples+num_samples] += samples * attenuation ** (i)
            
        delayed_signal = delayed_signal[:,:,:num_samples]
        
        samples = (samples + volume_factor * delayed_signal)/2

        return samples