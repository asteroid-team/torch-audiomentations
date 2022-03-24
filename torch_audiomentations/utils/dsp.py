import torch


def calculate_rms(samples):
    """
    Calculates the root mean square.

    Based on https://github.com/iver56/audiomentations/blob/master/audiomentations/core/utils.py
    """
    return torch.sqrt(torch.mean(torch.square(samples), dim=-1, keepdim=False))


def calculate_desired_noise_rms(clean_rms, snr):
    """
    Given the Root Mean Square (RMS) of a clean sound and a desired signal-to-noise ratio (SNR),
    calculate the desired RMS of a noise sound to be mixed in.
    Based on https://github.com/Sato-Kunihiko/audio-SNR/blob/8d2c933b6c0afe6f1203251f4877e7a1068a6130/create_mixed_audio_file.py#L20

    :param clean_rms: Root Mean Square (RMS) - a value between 0.0 and 1.0
    :param snr: Signal-to-Noise (SNR) Ratio in dB - typically somewhere between -20 and 60
    :return:
    """
    noise_rms = clean_rms / (10 ** (snr / 20))
    return noise_rms


def resample_audio(audio, orig_sr, target_sr):
    # TODO: We can probably remove this function and call resample directly where needed
    """Resamples the audio to a new sampling rate."""
    import librosa

    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def convert_decibels_to_amplitude_ratio(decibels):
    return 10 ** (decibels / 20)


def convert_amplitude_ratio_to_decibels(amplitude_ratio):
    return 20 * torch.log10(amplitude_ratio)
