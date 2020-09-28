from .dsp import resample_audio
import os
import glob
import soundfile


SUPPORTED_EXTENSIONS = [".wav"]


def find_audio_files(path):
    """Finds all audio files of supported extensions."""
    files = []

    for supported_extension in SUPPORTED_EXTENSIONS:
        files.extend(glob.glob(os.path.join(path, "*" + supported_extension)))

    return files


def load_audio(audio_file_path, sample_rate=None, start=0, stop=None):
    """Loads the audio given the path of an audio file."""
    audio, source_sample_rate = soundfile.read(audio_file_path, start=start, stop=stop)

    if sample_rate:
        audio = resample_audio(audio, source_sample_rate, sample_rate)

    return audio
