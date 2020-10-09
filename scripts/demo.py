import os
import random
from pathlib import Path

import librosa
import numpy as np
import time
import torch
from scipy.io import wavfile

from torch_audiomentations import PolarityInversion

SAMPLE_RATE = 16000

BASE_DIR = Path(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
SCRIPTS_DIR = BASE_DIR / "scripts"
TEST_FIXTURES_DIR = BASE_DIR / "test_fixtures"


class timer(object):
    """
    timer: A class used to measure the execution time of a block of code that is
    inside a "with" statement.

    Example:

    ```
    with timer("Count to 500000"):
        x = 0
        for i in range(500000):
            x += 1
        print(x)
    ```

    Will output:
    500000
    Count to 500000: 0.04 s

    Warning: The time resolution used here may be limited to 1 ms
    """

    def __init__(self, description="Execution time", verbose=False):
        self.description = description
        self.verbose = verbose
        self.execution_time = None

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.execution_time = time.time() - self.t
        if self.verbose:
            print("{}: {:.3f} s".format(self.description, self.execution_time))


if __name__ == "__main__":
    """
    For each transformation, apply it to an example sound and write the transformed sounds to
    an output folder. Also crudely measure and print execution time.
    """
    output_dir = os.path.join(SCRIPTS_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(42)
    random.seed(42)

    samples, _ = librosa.load(
        os.path.join(TEST_FIXTURES_DIR, "acoustic_guitar_0.wav"), sr=SAMPLE_RATE
    )

    transforms = [{"instance": PolarityInversion(p=1.0), "num_runs": 1}]

    execution_times = {}

    for transform in transforms:
        augmenter = transform["instance"]
        run_name = (
            transform.get("name")
            if transform.get("name")
            else transform["instance"].__class__.__name__
        )
        execution_times[run_name] = []
        for i in range(transform["num_runs"]):
            output_file_path = os.path.join(
                output_dir, "{}_{:03d}.wav".format(run_name, i)
            )
            with timer() as t:
                augmented_samples = augmenter(
                    samples=torch.from_numpy(samples), sample_rate=SAMPLE_RATE
                ).numpy()
            execution_times[run_name].append(t.execution_time)
            wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    for run_name in execution_times:
        if len(execution_times[run_name]) > 1:
            print(
                "{:<32} {:.3f} s (std: {:.3f} s)".format(
                    run_name,
                    np.mean(execution_times[run_name]),
                    np.std(execution_times[run_name]),
                )
            )
        else:
            print("{:<32} {:.3f} s".format(run_name, np.mean(execution_times[run_name])))
