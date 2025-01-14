import numpy as np
import torch
from cpuinfo import get_cpu_info
from scipy.signal import convolve as scipy_convolve
from tqdm import tqdm

from scripts.demo import TEST_FIXTURES_DIR, timer
from scripts.plot import show_horizontal_bar_chart
from torch_audiomentations.utils.convolution import convolve as torch_convolve
from torch_audiomentations.utils.io import Audio

if __name__ == "__main__":
    file_path = TEST_FIXTURES_DIR / "acoustic_guitar_0.wav"
    sample_rate = 48000
    audio = Audio(sample_rate, mono=True)
    samples = audio(file_path).numpy()
    ir_samples = audio(TEST_FIXTURES_DIR / "ir" / "impulse_response_0.wav").numpy()

    is_cuda_available = torch.cuda.is_available()
    print("Is torch CUDA available:", is_cuda_available)

    num_examples = 32

    execution_times = {
        # tuples of (description, batch size)
        ("scipy direct", 1): [],
        ("scipy FFT", 1): [],
        ("torch FFT CPU", 1): [],
        ("torch FFT CPU", num_examples): [],
        ("torch FFT CUDA", 1): [],
        ("torch FFT CUDA", num_examples): [],
    }

    for i in tqdm(range(num_examples), desc="scipy, method='direct', batch size=1"):
        with timer("scipy direct") as t:
            expected_output = scipy_convolve(samples, ir_samples, method="direct")
        execution_times[(t.description, 1)].append(t.execution_time)

    for i in tqdm(range(num_examples), desc="scipy, method='fft', batch size=1"):
        with timer("scipy FFT") as t:
            expected_output = scipy_convolve(samples, ir_samples, method="fft")
        execution_times[(t.description, 1)].append(t.execution_time)

    pytorch_samples_cpu = torch.from_numpy(samples)
    pytorch_ir_samples_cpu = torch.from_numpy(ir_samples)

    for i in tqdm(range(num_examples), desc="torch FFT CPU, batch size=1"):
        with timer("torch FFT CPU") as t:
            _ = torch_convolve(pytorch_samples_cpu, pytorch_ir_samples_cpu).numpy()
        execution_times[(t.description, 1)].append(t.execution_time)

    pytorch_samples_batch_cpu = torch.stack([pytorch_samples_cpu] * num_examples)

    for i in tqdm(range(5), desc="torch FFT CPU, batch size={}".format(num_examples)):
        with timer("torch FFT CPU") as t:
            _ = torch_convolve(pytorch_samples_batch_cpu, pytorch_ir_samples_cpu).numpy()
        execution_times[(t.description, num_examples)].append(t.execution_time)

    if is_cuda_available:
        pytorch_samples_cuda = torch.from_numpy(samples).cuda()
        pytorch_ir_samples_cuda = torch.from_numpy(ir_samples).cuda()

        for i in tqdm(range(num_examples), desc="torch FFT CUDA, batch size=1"):
            with timer("torch FFT CUDA") as t:
                _ = (
                    torch_convolve(pytorch_samples_cuda, pytorch_ir_samples_cuda)
                    .cpu()
                    .numpy()
                )
            execution_times[(t.description, 1)].append(t.execution_time)

        pytorch_samples_batch_cuda = pytorch_samples_batch_cpu.cuda()

        for i in tqdm(
            range(5), desc="torch FFT CUDA, batch size={}".format(num_examples)
        ):
            with timer("torch FFT CUDA") as t:
                _ = (
                    torch_convolve(pytorch_samples_batch_cuda, pytorch_ir_samples_cuda)
                    .cpu()
                    .numpy()
                )
            execution_times[(t.description, num_examples)].append(t.execution_time)

    normalized_execution_times = {}
    for description, batch_size in execution_times:
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
        normalized_execution_times[
            "{}, batch size={}".format(description, batch_size)
        ] = batch_execution_time

    cpu_info = get_cpu_info()
    cpu_info_string = "CPU: {}".format(cpu_info["brand_raw"])
    print(cpu_info_string)

    plot_title = "Convolving an IR with {} sounds\n{}".format(
        num_examples, cpu_info_string
    )

    if is_cuda_available:
        cuda_device_name = torch.cuda.get_device_name()
        cuda_device_string = "CUDA device: {}".format(cuda_device_name)
        print(cuda_device_string)
        plot_title += "\n{}".format(cuda_device_string)

    show_horizontal_bar_chart(normalized_execution_times, plot_title)
