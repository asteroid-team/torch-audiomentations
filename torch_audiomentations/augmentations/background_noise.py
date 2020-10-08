import random

import math
import numpy as np
import soundfile
import torch

from ..core.transforms_interface import BaseWaveformTransform, EmptyPathException
from ..utils.dsp import calculate_rms, calculate_desired_noise_rms
from ..utils.file import find_audio_files, load_audio


class ApplyBackgroundNoise(BaseWaveformTransform):
    """
    Applies a background noise to the input signal.
    """

    def __init__(
        self,
        bg_path,
        min_snr_in_db: float = 3.0,
        max_snr_in_db: float = 30.0,
        device=torch.device("cpu"),
        p: float = 0.5,
    ):
        # TODO: infer device from the given samples instead
        super(ApplyBackgroundNoise, self).__init__(p)
        self.bg_path = find_audio_files(bg_path)

        if len(self.bg_path) == 0:
            raise EmptyPathException("There are no supported audio files found.")

        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        self.device = device

    def randomize_parameters(self, selected_samples, sample_rate: int):
        if self.parameters["should_apply"] and selected_samples.size(0) > 0:
            bg_file_paths = random.choices(self.bg_path, k=selected_samples.size(0))
            bg_audios = []
            for index, bg_file_path in enumerate(bg_file_paths):
                bg_audio_info = soundfile.info(bg_file_path, verbose=True)
                bg_audio_num_samples = math.ceil(
                    sample_rate * bg_audio_info.frames / bg_audio_info.samplerate
                )
                samples_num_samples = selected_samples[index].size(0)

                # ensure that background noise has the same length as the sample
                if bg_audio_num_samples < samples_num_samples:
                    bg_start_index = 0
                    bg_stop_index = bg_audio_info.frames
                    loaded_bg_audio_num_samples = bg_audio_num_samples

                    current_bg_audio = load_audio(
                        bg_file_path,
                        sample_rate=sample_rate,
                        start=bg_start_index,
                        stop=bg_stop_index,
                    )
                    bg_audio = [current_bg_audio]

                    # TODO: Factor out this bit that repeats the audio until the desired length
                    #  has been reached, and then trims away any excess audio from the end
                    while loaded_bg_audio_num_samples < samples_num_samples:
                        current_bg_audio = bg_audio[-1]
                        loaded_bg_audio_num_samples += current_bg_audio.shape[0]
                        bg_audio.append(current_bg_audio)

                    bg_audio = np.concatenate(bg_audio)
                    bg_audio = bg_audio[:samples_num_samples]
                else:
                    factor = int(bg_audio_info.samplerate / sample_rate)
                    max_bg_offset = max(
                        0, bg_audio_info.frames - samples_num_samples * factor
                    )
                    bg_start_index = random.randint(0, max_bg_offset)
                    bg_stop_index = bg_start_index + samples_num_samples * factor
                    bg_audio = load_audio(
                        bg_file_path,
                        sample_rate=sample_rate,
                        start=bg_start_index,
                        stop=bg_stop_index,
                    )

                bg_audios.append(torch.from_numpy(bg_audio))

            self.parameters["snr_in_db"] = random.uniform(
                self.min_snr_in_db, self.max_snr_in_db
            )
            self.parameters["bg_audios"] = torch.stack(bg_audios)

    def apply_transform(self, selected_samples, sample_rate: int):
        selected_samples = selected_samples.to(self.device)
        bg_audios = self.parameters["bg_audios"].to(self.device)

        # calculate sample and background audio RMS
        samples_rms = calculate_rms(selected_samples)
        bg_audios_rms = calculate_rms(bg_audios)

        desired_bg_audios_rms = calculate_desired_noise_rms(
            samples_rms, self.parameters["snr_in_db"]
        )
        bg_audios = bg_audios * (desired_bg_audios_rms / bg_audios_rms)

        return selected_samples + bg_audios
