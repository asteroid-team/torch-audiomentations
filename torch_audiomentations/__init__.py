from .augmentations.background_noise import ApplyBackgroundNoise
from .augmentations.impulse_response import ApplyImpulseResponse
from .augmentations.gain import Gain
from .augmentations.polarity_inversion import PolarityInversion

# TODO: Revise these imports
from .utils.convolution import convolve
from .utils.dsp import calculate_rms, calculate_desired_noise_rms, resample_audio
from .utils.file import find_audio_files, load_audio

__version__ = "0.0.1"
