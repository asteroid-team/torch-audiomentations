import os
import random
import time
from pathlib import Path

import librosa
import numpy as np
import torch
from scipy.io import wavfile

from torch_audiomentations import (
    PolarityInversion,
    Gain,
    PeakNormalization,
    Compose,
    Shift,
    AddBackgroundNoise,
    ApplyImpulseResponse,
    AddColoredNoise,
    HighPassFilter,
    LowPassFilter,
    BandPassFilter,
    PitchShift,
    BandStopFilter,
    TimeInversion,
    Padding,
    VTLP,
)
from torch_audiomentations.augmentations.shuffle_channels import ShuffleChannels
from torch_audiomentations.core.transforms_interface import (
    ModeNotSupportedException,
    MultichannelAudioNotSupportedException,
)
from torch_audiomentations.utils.object_dict import ObjectDict

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
    output_dir = os.path.join(SCRIPTS_DIR, "demo_output")
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(43)
    np.random.seed(43)
    random.seed(43)

    batch = [["p286_011.wav"], ["perfect-alley1.ogg", "perfect-alley2.ogg"]]

    for filenames in batch:
        audios = []
        batch_sample_rate = None
        for filename in filenames:
            samples1, batch_sample_rate = librosa.load(
                os.path.join(TEST_FIXTURES_DIR, filename), sr=None, mono=False
            )
            if samples1.ndim == 1:
                samples1 = samples1.reshape((1, -1))
            audios.append(samples1)
        samples = np.stack(audios, axis=0)
        samples = torch.from_numpy(samples)

        modes = ["per_example"]
        if samples.shape[0] > 1:
            modes.append("per_batch")
        if samples.shape[1] > 1:
            modes.append("per_channel")
        for mode in modes:
            transforms = [
                {
                    "get_instance": lambda: AddBackgroundNoise(
                        background_paths=TEST_FIXTURES_DIR / "bg", mode=mode, p=1.0
                    ),
                    "num_runs": 5,
                },
                {
                    "get_instance": lambda: AddColoredNoise(mode=mode, p=1.0),
                    "num_runs": 5,
                },
                {
                    "get_instance": lambda: ApplyImpulseResponse(
                        ir_paths=TEST_FIXTURES_DIR / "ir", mode=mode, p=1.0
                    ),
                    "num_runs": 1,
                },
                {
                    "get_instance": lambda: ApplyImpulseResponse(
                        ir_paths=TEST_FIXTURES_DIR / "ir",
                        compensate_for_propagation_delay=True,
                        mode=mode,
                        p=1.0,
                    ),
                    "name": "ApplyImpulseResponse with compensate_for_propagation_delay set to True",
                    "num_runs": 1,
                },
                {"get_instance": lambda: BandPassFilter(mode=mode, p=1.0), "num_runs": 5},
                {"get_instance": lambda: BandStopFilter(mode=mode, p=1.0), "num_runs": 5},
                {
                    "get_instance": lambda: Compose(
                        transforms=[
                            Gain(
                                min_gain_in_db=-18.0,
                                max_gain_in_db=-16.0,
                                mode=mode,
                                p=1.0,
                            ),
                            PeakNormalization(mode=mode, p=1.0),
                        ],
                        shuffle=True,
                    ),
                    "name": "Shuffled Compose with Gain and PeakNormalization",
                    "num_runs": 5,
                },
                {
                    "get_instance": lambda: Compose(
                        transforms=[
                            Gain(
                                min_gain_in_db=-18.0,
                                max_gain_in_db=-16.0,
                                mode=mode,
                                p=0.5,
                            ),
                            PolarityInversion(mode=mode, p=0.5),
                        ],
                        shuffle=True,
                    ),
                    "name": "Compose with Gain and PolarityInversion",
                    "num_runs": 5,
                },
                {"get_instance": lambda: Gain(mode=mode, p=1.0), "num_runs": 5},
                {"get_instance": lambda: HighPassFilter(mode=mode, p=1.0), "num_runs": 5},
                {"get_instance": lambda: LowPassFilter(mode=mode, p=1.0), "num_runs": 5},
                {"get_instance": lambda: Padding(mode=mode, p=1.0), "num_runs": 5},
                {
                    "get_instance": lambda: PeakNormalization(mode=mode, p=1.0),
                    "num_runs": 1,
                },
                {
                    "get_instance": lambda: PitchShift(
                        sample_rate=batch_sample_rate, mode=mode, p=1.0
                    ),
                    "num_runs": 5,
                },
                {
                    "get_instance": lambda: PolarityInversion(mode=mode, p=1.0),
                    "num_runs": 1,
                },
                {"get_instance": lambda: Shift(mode=mode, p=1.0), "num_runs": 5},
                {
                    "get_instance": lambda: ShuffleChannels(mode=mode, p=1.0),
                    "num_runs": 5,
                },
                {"get_instance": lambda: VTLP(mode=mode, p=1.0), "num_runs": 5},
                {"get_instance": lambda: TimeInversion(mode=mode, p=1.0), "num_runs": 1},
            ]

            execution_times = {}

            for transform in transforms:
                try:
                    augmenter = transform["get_instance"]()
                except ModeNotSupportedException:
                    continue
                transform_name = (
                    transform.get("name")
                    if transform.get("name")
                    else augmenter.__class__.__name__
                )
                execution_times[transform_name] = []
                for i in range(transform["num_runs"]):
                    with timer() as t:
                        try:
                            augmented_samples = augmenter(
                                samples=samples, sample_rate=batch_sample_rate
                            )
                        except MultichannelAudioNotSupportedException as e:
                            print(e)
                            continue
                        print(
                            augmenter.__class__.__name__,
                            "is output ObjectDict:",
                            type(augmented_samples) is ObjectDict,
                        )
                        augmented_samples = (
                            augmented_samples.samples.numpy()
                            if type(augmented_samples) is ObjectDict
                            else augmented_samples.numpy()
                        )
                    execution_times[transform_name].append(t.execution_time)
                    for example_idx, original_filename in enumerate(filenames):
                        output_file_path = os.path.join(
                            output_dir,
                            "{}_{}_{:03d}_{}.wav".format(
                                transform_name, mode, i, Path(original_filename).stem
                            ),
                        )
                        wavfile.write(
                            output_file_path,
                            rate=batch_sample_rate,
                            data=augmented_samples[example_idx].transpose(),
                        )

            for transform_name in execution_times:
                if len(execution_times[transform_name]) > 1:
                    print(
                        "{:<52} {:.3f} s (std: {:.3f} s)".format(
                            transform_name,
                            np.mean(execution_times[transform_name]),
                            np.std(execution_times[transform_name]),
                        )
                    )
                else:
                    print(
                        "{:<52} {:.3f} s".format(
                            transform_name, np.mean(execution_times[transform_name])
                        )
                    )
