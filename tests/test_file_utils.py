import unittest
from pathlib import Path

from tests.utils import TEST_FIXTURES_DIR
from torch_audiomentations.utils.file import find_audio_files


class TestFileUtils(unittest.TestCase):
    def test_find_audio_files(self):
        file_paths = find_audio_files(TEST_FIXTURES_DIR)
        file_paths = [Path(fp).name for fp in file_paths]
        self.assertEqual(
            set(file_paths),
            {
                "acoustic_guitar_0.wav",
                "bg.wav",
                "bg_short.WAV",
                "impulse_response_0.wav",
                "stereo_noise.wav",
            },
        )
