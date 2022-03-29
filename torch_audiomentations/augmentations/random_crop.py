import torch
import typing
import warnings
from torch_audiomentations.utils.multichannel import is_multichannel
from ..core.transforms_interface import MultichannelAudioNotSupportedException

class RandomCrop(torch.nn.Module):

    supports_multichannel = True

    def __init__(
        self,
        seconds: float,
        sampling_rate: int
    ):
        super(RandomCrop,self).__init__()
        self.sampling_rate = sampling_rate
        self.num_samples = int(self.sampling_rate * seconds)

    
    def forward(self, samples, sampling_rate: typing.Optional[int] = None):
        
        sample_rate = sampling_rate or self.sampling_rate
        if sample_rate is None:
            raise RuntimeError("sample_rate is required")

        if len(samples) == 0:
            warnings.warn(
                "An empty samples tensor was passed to {}".format(self.__class__.__name__)
            )
            return samples

        if len(samples.shape) != 3:
            raise RuntimeError(
                "torch-audiomentations expects input tensors to be three-dimensional, with"
                " dimension ordering like [batch_size, num_channels, num_samples]. If your"
                " audio is mono, you can use a shape like [batch_size, 1, num_samples]."
            )

        if is_multichannel(samples):
            if samples.shape[1] > samples.shape[2]:
                warnings.warn(
                    "Multichannel audio must have channels first, not channels last. In"
                    " other words, the shape must be (batch size, channels, samples), not"
                    " (batch_size, samples, channels)"
                )
            if not self.supports_multichannel:
                raise MultichannelAudioNotSupportedException(
                    "{} only supports mono audio, not multichannel audio".format(
                        self.__class__.__name__
                    )
                )


        if samples.shape[2] < self.num_samples:
            warnings.warn("audio length less than cropping length")
            return samples
            
        
        start_indices = torch.randint(0,samples.shape[2] - self.num_samples,(samples.shape[2],))
        samples_cropped = torch.empty((samples.shape[0],samples.shape[1],self.num_samples))
        for i,sample in enumerate(samples):
            
            samples_cropped[i] = sample.unsqueeze(0)[:,:,start_indices[i]:start_indices[i]+self.num_samples]
        
        return samples_cropped






