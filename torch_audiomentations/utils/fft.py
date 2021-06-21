try:
    # This works in PyTorch>=1.7
    from torch.fft import irfft, rfft
except ModuleNotFoundError:
    # PyTorch<=1.6
    raise Exception(
        "torch-audiomentations does not pytorch<=1.6. Please upgrade to pytorch 1.7 or newer.",
    )
