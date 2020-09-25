import numpy as np
import torch


def calculate_rms(samples):
    """
    Calculates the root mean square.

		Taken from https://github.com/iver56/audiomentations/blob/master/audiomentations/core/utils.py
    """
    return torch.sqrt(torch.mean(torch.square(samples), -1))

def calculate_desired_noise_rms(clean_rms, snr):
    """
    Given the Root Mean Square (RMS) of a clean sound and a desired signal-to-noise ratio (SNR),
    calculate the desired RMS of a noise sound to be mixed in.
    Based on https://github.com/Sato-Kunihiko/audio-SNR/blob/8d2c933b6c0afe6f1203251f4877e7a1068a6130/create_mixed_audio_file.py#L20
		Taken from https://github.com/iver56/audiomentations/blob/master/audiomentations/core/utils.py
    :param clean_rms: Root Mean Square (RMS) - a value between 0.0 and 1.0
    :param snr: Signal-to-Noise (SNR) Ratio in dB - typically somewhere between -20 and 60
    :return:
    """
    a = float(snr) / 20
    noise_rms = clean_rms / (10 ** a)
    return noise_rms
