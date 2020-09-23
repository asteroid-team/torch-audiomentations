import numpy as np
import random
import torch
from torch_audiomentations.core.transforms_interface import BasicTransform
from torch_audiomentations.utils.convolution import convolve
from torch_audiomentations.utils.file import find_files, load_audio


class ImpulseResponse(BasicTransform):
    """
    Convolves an input signal with an impulse response.
    """

    def __init__(self, ir_path, convolve_mode="full", p=0.5):
        super(ImpulseResponse, self).__init__(p)
        self.ir_path = find_files(ir_path)
        self.convolve_mode = convolve_mode

    def randomize_parameters(self, samples, sample_rate):
        super(ImpulseResponse, self).randomize_parameters(samples, sample_rate)

        if self.parameters["should_apply"]:
            self.parameters["ir_file_path"] = random.choice(self.ir_path)

    def apply(self, samples, sample_rate):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ir = torch.from_numpy(load_audio(self.parameters["ir_file_path"], sample_rate)).to(device)

        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples).to(device)
        else:
            samples = samples.to(device)

        signal_ir = convolve(samples, ir, mode=self.convolve_mode)
        signal_ir = signal_ir.cpu().numpy()
        return signal_ir

