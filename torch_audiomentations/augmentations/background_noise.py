import random
import torch
from ..core.transforms_interface import BasicTransform
from ..utils.file import find_audio_files, load_audio
from ..utils.dsp import calculate_rms, calculate_desired_noise_rms


class ApplyBackgroundNoise(BasicTransform):
    """
    Applies a background noise to the input signal.
    """

    def __init__(self, bg_path, snr_in_db, device=torch.device("cpu"), p=0.5):
        super(ApplyBackgroundNoise, self).__init__(p)
        self.bg_path = find_audio_files(bg_path)
        assert len(self.bg_path) > 0
        self.snr_in_db = snr_in_db
        self.device = device

    def randomize_parameters(self, samples, sample_rate):
        super(ApplyBackgroundNoise, self).randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"] and samples.size(0) > 0:
            bg_file_paths = random.choices(self.bg_path, k=samples.size(0))
            bg_audios = []
            for index, bg_file_path in enumerate(bg_file_paths):
                bg_audio = torch.from_numpy(load_audio(bg_file_path, sample_rate))
                max_bg_offset = max(0, bg_audio.size(0) - samples[index].size(0) - 1)

                bg_start_index = random.randint(0, max_bg_offset)
                bg_end_index = bg_start_index + samples[index].size(0)
                bg_audios.append(bg_audio[bg_start_index:bg_end_index])

            self.parameters["snr_in_db"] = self.snr_in_db
            self.parameters["bg_audios"] = torch.stack(bg_audios)

    def apply(self, samples, sample_rate):
        samples = samples.to(self.device)
        bg_audios = self.parameters["bg_audios"].to(self.device)

        # calculate sample and background audio RMS
        samples_rms = calculate_rms(samples)
        bg_audios_rms = calculate_rms(bg_audios)

        desired_bg_audios_rms = calculate_desired_noise_rms(
            samples_rms, self.parameters["snr_in_db"]
        )
        bg_audios = bg_audios * (desired_bg_audios_rms / bg_audios_rms)

        while bg_audios.size(1) < samples.size(1):
            bg_audios = bg_audios.repeat(1, 2)

        if bg_audios.size(1) > samples.size(1):
            bg_audios = bg_audios[:, : samples.size(1)]

        return samples + bg_audios
