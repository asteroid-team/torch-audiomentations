import numpy as np
import random
import torch
from ..core.transforms_interface import BasicTransform
from ..utils.convolution import convolve
from ..utils.file import find_audio_files, load_audio


class ApplyImpulseResponse(BasicTransform):
    """
    Convolves an input signal with an impulse response.
    """

    def __init__(self, ir_path, convolve_mode="full", p=0.5):
        super(ApplyImpulseResponse, self).__init__(p)
        self.ir_path = find_audio_files(ir_path)
        self.convolve_mode = convolve_mode

    def randomize_parameters(self, samples, sample_rate):
        super(ApplyImpulseResponse, self).randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["ir_file_path"] = random.choice(self.ir_path)

    def apply(self, samples, sample_rate):
        ir = torch.from_numpy(load_audio(self.parameters["ir_file_path"], sample_rate))
        return convolve(samples, ir, mode=self.convolve_mode)
