import os
import glob
import librosa


SUPPORTED_EXTENSIONS = ['.wav']


def find_files(path):
    """Finds all files of supported extensions."""
    files = []

    for supported_extension in SUPPORTED_EXTENSIONS:
        files.extend(glob.glob(os.path.join(path, '*' + supported_extension)))

    return files

def load_audio(audio_file_path, sample_rate):
    """Loads the audio given the path of an audio file."""
    return librosa.load(audio_file_path, sr=sample_rate)[0]

