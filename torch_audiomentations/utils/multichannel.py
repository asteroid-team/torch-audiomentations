def is_multichannel(samples):
    """

    :param samples:
    :return:
    """
    return len(samples.shape) > 2 and samples.shape[1] > 1
