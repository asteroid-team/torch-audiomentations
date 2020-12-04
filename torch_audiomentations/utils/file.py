import glob
from pathlib import Path

import soundfile

from .dsp import resample_audio

SUPPORTED_EXTENSIONS = [".wav"]


def find_audio_files(path):
    """Finds all audio files of supported extensions in the given path."""
    files = []

    for supported_extension in SUPPORTED_EXTENSIONS:
        files.extend(
            glob.glob(
                str(Path(path) / "**" / ("*" + supported_extension)), recursive=True
            )
        )

    return files


def load_audio(audio_file_path, sample_rate=None, start=0, stop=None):
    # TODO: Clarify whether start/stop is in samples or in seconds, and whether or not it
    #  relates to the original or the resampled audio.
    """Loads the audio given the path of an audio file."""
    audio, source_sample_rate = soundfile.read(audio_file_path, start=start, stop=stop)

    if sample_rate:
        audio = resample_audio(audio, source_sample_rate, sample_rate)

    # TODO: return sample rate as well
    return audio
