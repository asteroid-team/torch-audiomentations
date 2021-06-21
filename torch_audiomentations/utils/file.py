import os
from pathlib import Path

import soundfile

from .dsp import resample_audio

SUPPORTED_EXTENSIONS = (".wav",)


def find_audio_files(
    root_path,
    filename_endings=SUPPORTED_EXTENSIONS,
    traverse_subdirectories=True,
    follow_symlinks=True,
):
    """Return a list of paths to all audio files with the given extension(s) in a directory.
    Also traverses subdirectories by default.
    """
    file_paths = []

    for root, dirs, filenames in os.walk(root_path, followlinks=follow_symlinks):
        filenames = sorted(filenames)
        for filename in filenames:
            input_path = os.path.abspath(root)
            file_path = os.path.join(input_path, filename)

            if filename.lower().endswith(filename_endings):
                file_paths.append(Path(file_path))
        if not traverse_subdirectories:
            # prevent descending into subfolders
            break

    return file_paths


def load_audio(audio_file_path, sample_rate=None, start=0, stop=None):
    # TODO: Clarify whether start/stop is in samples or in seconds, and whether or not it
    #  relates to the original or the resampled audio.
    """Loads the audio given the path of an audio file."""
    audio, source_sample_rate = soundfile.read(audio_file_path, start=start, stop=stop)

    if sample_rate:
        audio = resample_audio(audio, source_sample_rate, sample_rate)

    # TODO: return sample rate as well
    return audio
