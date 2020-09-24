def is_multichannel(samples):
    """

    :param samples:
    :return:
    """
    return len(samples.shape) > 2
