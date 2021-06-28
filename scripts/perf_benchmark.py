import seaborn as sns
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import torch
from tqdm import tqdm

from torch_audiomentations import PolarityInversion, Gain, PeakNormalization, Shift, ShuffleChannels, LowPassFilter, HighPassFilter

BASE_DIR = Path(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
SCRIPTS_DIR = BASE_DIR / "scripts"


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


def measure_execution_time(
    transform, batch_size, num_channels, duration, sample_rate, device_name, device
):
    transform_name = transform.__class__.__name__
    num_samples = int(duration * sample_rate)
    samples = torch.rand(
        (batch_size, num_channels, num_samples), dtype=torch.float32, device=device
    )
    perf_objects = []
    for i in range(3):
        perf_obj = {
            "metrics": {},
            "params": {
                "transform": transform_name,
                "batch_size": batch_size,
                "num_channels": num_channels,
                "duration": duration,
                "sample_rate": sample_rate,
                "num_samples": num_samples,
                "device_name": device_name,
            },
        }
        with timer() as t:
            transform(samples=samples, sample_rate=sample_rate).cpu()
        perf_obj["metrics"]["execution_time"] = t.execution_time
        perf_objects.append(perf_obj)
    return perf_objects


if __name__ == "__main__":
    """
    For each transformation, apply it to an example sound and write the transformed sounds to
    an output folder. Also crudely measure and print execution time.
    """
    np.random.seed(42)
    random.seed(42)

    output_dir = SCRIPTS_DIR / "perf_benchmark_output"
    os.makedirs(output_dir, exist_ok=True)

    params = dict(
        batch_sizes=[1, 2, 4, 8, 16],
        channels=[1, 2, 4, 8],
        durations=[1, 2, 4, 8, 16],
        sample_rates=[16000],
        devices=["cpu", "cuda"],
    )

    if not torch.cuda.is_available():
        params["devices"].remove("cuda")

    devices = {
        device_name: torch.device(device_name) for device_name in params["devices"]
    }

    transforms = [
        Gain(p=1.0),
        HighPassFilter(p=1.0),
        LowPassFilter(p=1.0),
        PolarityInversion(p=1.0),
        PeakNormalization(p=1.0),
        Shift(p=1.0),
        ShuffleChannels(p=1.0),
    ]

    perf_objects = []

    for device_name in params["devices"]:
        device = devices[device_name]
        for batch_size in tqdm(params["batch_sizes"]):
            for num_channels in params["channels"]:
                for duration in params["durations"]:
                    for sample_rate in params["sample_rates"]:
                        for transform in transforms:
                            perf_objects += measure_execution_time(
                                transform,
                                batch_size,
                                num_channels,
                                duration,
                                sample_rate,
                                device_name,
                                device,
                            )

    params_to_group_by = ["batch_size", "num_channels", "num_samples", "device_name"]
    for group_by_param in tqdm(params_to_group_by, desc="Making plots"):
        param_values = []
        metric_values = []
        transform_names = []
        for perf_obj in perf_objects:
            param_values.append(perf_obj["params"][group_by_param])
            metric_values.append(perf_obj["metrics"]["execution_time"])
            transform_names.append(perf_obj["params"]["transform"])

        df = pd.DataFrame(
            {
                group_by_param: param_values,
                "exec_time": metric_values,
                "transform": transform_names,
            }
        )

        violin_plot_file_path = str(output_dir / "{}_plot.png".format(group_by_param))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("execution time grouped by {}".format(group_by_param))
        g = sns.boxplot(x=group_by_param, y="exec_time", data=df, ax=ax, hue="transform")
        g.set_yscale("log")
        fig.tight_layout()
        plt.savefig(violin_plot_file_path)
        plt.close(fig)
