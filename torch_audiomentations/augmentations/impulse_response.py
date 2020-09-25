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
        assert(len(self.ir_path) > 0)
        self.convolve_mode = convolve_mode

    def randomize_parameters(self, samples, sample_rate):
        super(ApplyImpulseResponse, self).randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"] and samples.size(0) > 0:
            ir_paths = random.choices(self.ir_path, k=samples.size(0))
            ir_sounds = []
            max_ir_sound_length = 0
            for ir_path in ir_paths:
                ir_samples = load_audio(ir_path, sample_rate)
                num_samples = len(ir_samples)
                if num_samples > max_ir_sound_length:
                    max_ir_sound_length = num_samples
                ir_samples = torch.from_numpy(ir_samples)
                ir_sounds.append(ir_samples)
            for i in range(len(ir_sounds)):
                placeholder = torch.zeros(size=(max_ir_sound_length,), dtype=torch.float32)
                placeholder[0 : len(ir_sounds[i])] = ir_sounds[i]
                ir_sounds[i] = placeholder
            self.parameters["ir_sounds"] = torch.stack(ir_sounds)

    def apply(self, samples, sample_rate):
        ir = self.parameters["ir_sounds"]
        return convolve(samples, ir, mode=self.convolve_mode)
