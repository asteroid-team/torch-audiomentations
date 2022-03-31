import os
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest

from tests.utils import TEST_FIXTURES_DIR
from torch_audiomentations.utils.file import find_audio_files, find_audio_files_in_paths


class TestFileUtils:
    def test_find_audio_files(self):
        file_paths = find_audio_files(TEST_FIXTURES_DIR)
        file_paths = [Path(fp).name for fp in file_paths]
        assert set(file_paths) == {
            "acoustic_guitar_0.wav",
            "bg.wav",
            "bg_short.WAV",
            "impulse_response_0.wav",
            "stereo_noise.wav",
        }

    def test_find_audio_files_in_paths(self):
        paths = [
            os.path.join(TEST_FIXTURES_DIR, "bg"),
            os.path.join(TEST_FIXTURES_DIR, "bg_short"),
            os.path.join(TEST_FIXTURES_DIR, "ir", "impulse_response_0.wav"),
        ]
        file_paths = find_audio_files_in_paths(paths)
        file_paths = [Path(fp).name for fp in file_paths]
        assert set(file_paths) == {
            "bg.wav",
            "bg_short.WAV",
            "impulse_response_0.wav",
            "stereo_noise.wav",
        }

    @pytest.mark.skipif(
        os.name == "nt", reason="Symlink testing is not relevant on Windows"
    )
    def test_follow_directory_symlink(self):
        tmp_dir_1 = os.path.join(tempfile.gettempdir(), str(uuid.uuid4())[0:12])
        tmp_dir_2 = os.path.join(tempfile.gettempdir(), str(uuid.uuid4())[0:12])
        os.makedirs(tmp_dir_1, exist_ok=True)
        os.makedirs(tmp_dir_2, exist_ok=True)

        assert tmp_dir_1 != tmp_dir_2

        tmp_file_path = os.path.join(tmp_dir_1, "{}.wav".format(str(uuid.uuid4())[0:12]))
        shutil.copyfile(TEST_FIXTURES_DIR / "acoustic_guitar_0.wav", tmp_file_path)

        file_paths = find_audio_files(tmp_dir_2)
        assert len(file_paths) == 0

        symlink_path = os.path.join(tmp_dir_2, "subdir")
        os.symlink(tmp_dir_1, symlink_path, target_is_directory=True)

        file_paths = find_audio_files(tmp_dir_2)
        assert len(file_paths) == 1
        assert Path(file_paths[0]).name == Path(tmp_file_path).name

        os.unlink(symlink_path)
        os.unlink(tmp_file_path)
        os.rmdir(tmp_dir_1)
        os.rmdir(tmp_dir_2)
