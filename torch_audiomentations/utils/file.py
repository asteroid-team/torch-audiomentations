import os
from pathlib import Path
from typing import List, Union


SUPPORTED_EXTENSIONS = (".wav",)


def find_audio_files_in_paths(
    paths: Union[List[Path], List[str], Path, str],
    filename_endings=SUPPORTED_EXTENSIONS,
    traverse_subdirectories=True,
    follow_symlinks=True,
):
    """Return a list of paths to all audio files with the given extension(s) contained in the list or in its directories.
    Also traverses subdirectories by default.
    """

    file_paths = []

    if isinstance(paths, (list, tuple, set)):
        paths = list(paths)
    else:
        paths = [paths]

    for p in paths:
        if str(p).lower().endswith(SUPPORTED_EXTENSIONS):
            file_path = Path(os.path.abspath(p))
            file_paths.append(file_path)
        elif os.path.isdir(p):
            file_paths += find_audio_files(
                p,
                filename_endings=filename_endings,
                traverse_subdirectories=traverse_subdirectories,
                follow_symlinks=follow_symlinks,
            )
    return file_paths


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
