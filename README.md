# torch-audiomentations
Audio data augmentation in PyTorch. Inspired by [audiomentations](https://github.com/iver56/audiomentations).

# Setup

`pip install git+https://github.com/asteroid-team/torch-audiomentations`

Note: torch-audiomentations will be published on PyPI for easier installation later.

# Contribute

Contributors welcome! 
[Join the Asteroid's slack](https://join.slack.com/t/asteroid-dev/shared_invite/zt-cn9y85t3-QNHXKD1Et7qoyzu1Ji5bcA)
to start discussing about `torch-audiomentations` with us.

# Motivation: Speed

We don't want data augmentation to be a bottle neck in model training speed. Here is a
comparison of the time it takes to run 1D convolution:

![Convolve execution times](images/convolve_exec_time_plot.png)

# Current state

torch-audiomentations is in a very early development stage, so it's not ready for prime time yet.
Meanwhile, star the repo and stay tuned!

# Development

## Setup

A GPU-enabled development environment for torch-audiomentations can be created with conda:

* `conda create --name torch-audiomentations python=3.7.3`
* `conda activate torch-audiomentations`
* `conda install pytorch cudatoolkit=10.1 -c pytorch`
* `conda env update`

## Run tests

`pytest`

## Conventions

* Format python code with [black](https://github.com/psf/black)
* Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#381-docstrings)
* Use explicit relative imports, not absolute imports

# Acknowledgements

The development of torch-audiomentations is kindly backed by [Nomono](https://nomono.co/)
