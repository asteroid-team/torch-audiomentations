import torch

_NEXT_FAST_LEN = {}


def next_fast_len(size):
    """
    Returns the next largest number ``n >= size`` whose prime factors are all
    2, 3, or 5. These sizes are efficient for fast fourier transforms.
    Equivalent to :func:`scipy.fftpack.next_fast_len`.

    Note: This function was originally copied from the https://github.com/pyro-ppl/pyro
    repository, where the license was Apache 2.0. Any modifications to the original code can be
    found at https://github.com/asteroid-team/torch-audiomentations/commits

    :param int size: A positive number.
    :returns: A possibly larger number.
    :rtype int:
    """
    try:
        return _NEXT_FAST_LEN[size]
    except KeyError:
        pass

    assert isinstance(size, int) and size > 0
    next_size = size
    while True:
        remaining = next_size
        for n in (2, 3, 5):
            while remaining % n == 0:
                remaining //= n
        if remaining == 1:
            _NEXT_FAST_LEN[size] = next_size
            return next_size
        next_size += 1


def _complex_mul(a, b):
    """

    Note: This function was originally copied from the https://github.com/pyro-ppl/pyro
    repository, where the license was Apache 2.0. Any modifications to the original code can be
    found at https://github.com/asteroid-team/torch-audiomentations/commits

    :param a:
    :param b:
    :return:
    """

    ar, ai = a.unbind(-1)
    br, bi = b.unbind(-1)
    return torch.stack([ar * br - ai * bi, ar * bi + ai * br], dim=-1)


def convolve(signal, kernel, mode="full", method="fft"):
    """
    Computes the 1-d convolution of signal by kernel using FFTs.
    The two arguments should have the same rightmost dim, but may otherwise be
    arbitrarily broadcastable.

    Note: This function was originally copied from the https://github.com/pyro-ppl/pyro
    repository, where the license was Apache 2.0. Any modifications to the original code can be
    found at https://github.com/asteroid-team/torch-audiomentations/commits

    :param torch.Tensor signal: A signal to convolve.
    :param torch.Tensor kernel: A convolution kernel.
    :param str mode: One of: 'full', 'valid', 'same'.
    :return: A tensor with broadcasted shape. Letting ``m = signal.size(-1)``
        and ``n = kernel.size(-1)``, the rightmost size of the result will be:
        ``m + n - 1`` if mode is 'full';
        ``max(m, n) - min(m, n) + 1`` if mode is 'valid'; or
        ``max(m, n)`` if mode is 'same'.
    :rtype torch.Tensor:
    """
    if method != "fft":
        raise NotImplementedError('Only method="fft" is supported')

    m = signal.size(-1)
    n = kernel.size(-1)
    if mode == "full":
        truncate = m + n - 1
    elif mode == "valid":
        truncate = max(m, n) - min(m, n) + 1
    elif mode == "same":
        truncate = max(m, n)
    else:
        raise ValueError("Unknown mode: {}".format(mode))

    # Compute convolution using fft.
    padded_size = m + n - 1
    # Round up for cheaper fft.
    fast_ftt_size = next_fast_len(padded_size)
    f_signal = torch.rfft(
        torch.nn.functional.pad(signal, (0, fast_ftt_size - m)), 1, onesided=False
    )
    f_kernel = torch.rfft(
        torch.nn.functional.pad(kernel, (0, fast_ftt_size - n)), 1, onesided=False
    )
    f_result = _complex_mul(f_signal, f_kernel)
    result = torch.irfft(f_result, 1, onesided=False)

    start_idx = (padded_size - truncate) // 2
    return result[..., start_idx : start_idx + truncate]
