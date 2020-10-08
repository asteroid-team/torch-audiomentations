from .augmentations.polarity_inversion import PolarityInversion
from .augmentations.impulse_response import ApplyImpulseResponse
from .augmentations.background_noise import ApplyBackgroundNoise

# TODO: Revise these imports
from .utils.convolution import convolve
from .utils.file import find_audio_files, load_audio
from .utils.dsp import calculate_rms, calculate_desired_noise_rms, resample_audio

__version__ = "0.0.1"
