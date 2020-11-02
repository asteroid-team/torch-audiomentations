from .augmentations.gain import Gain
from .augmentations.polarity_inversion import PolarityInversion
from .augmentations.peak_normalization import PeakNormalization

from .utils.convolution import convolve
from .utils.config import from_dict, from_yaml

__version__ = "0.3.0"
