import pprint
from cpuinfo import get_cpu_info
import platform

import librosa
import numpy as np
import torch
from scipy.signal import convolve as scipy_convolve
from tqdm import tqdm

from scripts.demo import TEST_FIXTURES_DIR, timer
from torch_audiomentations.utils.convolution import convolve as torch_convolve

if __name__ == "__main__":
    file_path = TEST_FIXTURES_DIR / "acoustic_guitar_0.wav"
    sample_rate = 48000
    samples, _ = librosa.load(file_path, sr=sample_rate)
    ir_samples, _ = librosa.load(
        TEST_FIXTURES_DIR / "ir" / "impulse_response_0.wav", sr=sample_rate
    )

    is_cuda_available = torch.cuda.is_available()
    print("Is torch CUDA available:", is_cuda_available)

    num_examples = 32

    execution_times = {
        # tuples of (description, batch size)
        ("scipy direct", 1): [],
        ("scipy fft", 1): [],
        ("torch fft CPU", 1): [],
        ("torch fft CPU", num_examples): [],
        ("torch fft CUDA", 1): [],
        ("torch fft CUDA", num_examples): [],
    }

    for i in tqdm(range(num_examples), desc="scipy, method='direct', batch size=1"):
        with timer("scipy direct") as t:
            expected_output = scipy_convolve(samples, ir_samples, method="direct")
        execution_times[(t.description, 1)].append(t.execution_time)

    for i in tqdm(range(num_examples), desc="scipy, method='fft', batch size=1"):
        with timer("scipy fft") as t:
            expected_output = scipy_convolve(samples, ir_samples, method="fft")
        execution_times[(t.description, 1)].append(t.execution_time)

    pytorch_samples_cpu = torch.from_numpy(samples)
    pytorch_ir_samples_cpu = torch.from_numpy(ir_samples)

    for i in tqdm(range(num_examples), desc="torch fft CPU, batch size=1"):
        with timer("torch fft CPU") as t:
            _ = torch_convolve(pytorch_samples_cpu, pytorch_ir_samples_cpu).numpy()
        execution_times[(t.description, 1)].append(t.execution_time)

    pytorch_samples_batch_cpu = torch.stack([pytorch_samples_cpu] * num_examples)

    for i in tqdm(range(5), desc="torch fft CPU, batch size={}".format(num_examples)):
        with timer("torch fft CPU") as t:
            _ = torch_convolve(
                pytorch_samples_batch_cpu, pytorch_ir_samples_cpu
            ).numpy()
        execution_times[(t.description, num_examples)].append(t.execution_time)

    if is_cuda_available:
        pytorch_samples_cuda = torch.from_numpy(samples).cuda()
        pytorch_ir_samples_cuda = torch.from_numpy(ir_samples).cuda()

        for i in tqdm(range(num_examples), desc="torch fft CUDA, batch size=1"):
            with timer("torch fft CUDA") as t:
                _ = (
                    torch_convolve(pytorch_samples_cuda, pytorch_ir_samples_cuda)
                    .cpu()
                    .numpy()
                )
            execution_times[(t.description, 1)].append(t.execution_time)

        pytorch_samples_batch_cuda = pytorch_samples_batch_cpu.cuda()

        for i in tqdm(
            range(5), desc="torch fft CUDA, batch size={}".format(num_examples)
        ):
            with timer("torch fft CUDA") as t:
                _ = (
                    torch_convolve(pytorch_samples_batch_cuda, pytorch_ir_samples_cuda)
                    .cpu()
                    .numpy()
                )
            execution_times[(t.description, num_examples)].append(t.execution_time)

    for (description, batch_size) in execution_times:
        times = execution_times[(description, batch_size)]
        if len(times) == 0:
            continue
        times[0] = float(np.median(times))
        # We consider the first execution to be a warm-up, as it may take many magnitudes longer
        # Therefore, we simply replace its value with the median
        batch_execution_time = num_examples * sum(times) / (batch_size * len(times))
        print(
            "{:<20} batch size = {:<4} {:.3f} s".format(
                description, batch_size, batch_execution_time
            )
        )

    cpu_info = get_cpu_info()
    print("CPU: {}".format(cpu_info["brand_raw"]))

    if is_cuda_available:
        cuda_device_name = torch.cuda.get_device_name()
        print("CUDA device: {}".format(cuda_device_name))
