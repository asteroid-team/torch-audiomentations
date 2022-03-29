import torch
import typing

class RandomCrop(torch.nn.Module):

    requires_sample_rate = True

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

        if samples.shape[2] < self.num_samples:
            self.num_samples = samples.shape[2]
            raise RuntimeWarning("audio length less than cropping length")
            
        
        start_indices = torch.randint(0,samples.shape[2] - self.num_samples,(samples.shape[2],))
        samples_cropped = torch.empty((samples.shape[0],samples.shape[1],self.num_samples))
        for i,sample in enumerate(samples):
            
            samples_cropped[i] = sample.unsqueeze(0)[:,:,start_indices[i]:start_indices[i]+self.num_samples]
        
        return samples_cropped






