import torch


def convert_frequencies_to_mels(f: torch.Tensor) -> torch.Tensor:
    """
    Convert f hertz to m mels

    https://en.wikipedia.org/wiki/Mel_scale#Formula
    """
    return 2595.0 * torch.log10(1.0 + f / 700.0)


def convert_mels_to_frequencies(m: torch.Tensor) -> torch.Tensor:
    """
    Convert m mels to f hertz

    https://en.wikipedia.org/wiki/Mel_scale#History_and_other_formulas
    """
    return 700.0 * (10 ** (m / 2595.0) - 1.0)
