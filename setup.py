import codecs
import os
import re

from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="torch-audiomentations",
    version=find_version("torch_audiomentations", "__init__.py"),
    author="Iver Jordal",
    description="A Pytorch library for audio data augmentation. Inspired by audiomentations."
    " Useful for deep learning.",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/asteroid-team/torch-audiomentations",
    packages=find_packages(
        exclude=["build", "scripts", "dist", "images", "test_fixtures", "tests"]
    ),
    install_requires=[
        "julius>=0.2.3,<0.3",
        "librosa>=0.6.0",
        "torch>=1.7.0",
        "torchaudio>=0.7.0",
        "torch-pitch-shift>=1.2.0",
    ],
    extras_require={"extras": ["PyYAML"]},
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
