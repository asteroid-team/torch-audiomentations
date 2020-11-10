from .augmentations.gain import Gain
from .augmentations.peak_normalization import PeakNormalization
from .augmentations.polarity_inversion import PolarityInversion
from .core.composition import Compose
from .utils.convolution import convolve
from .utils.config import from_dict, from_yaml

__version__ = "0.4.0"
