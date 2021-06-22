# TODO: WRITE SOME REAL TESTS

import torch
from torch_audiomentations import Compose, PitchShift


# Initialize augmentation callable
apply_augmentation_per_example = Compose(
    transforms=[
        PitchShift(16000, p=1, mode="per_example"),
    ]
)
apply_augmentation_per_channel = Compose(
    transforms=[
        PitchShift(16000, p=1, mode="per_channel"),
    ]
)
apply_augmentation_per_batch = Compose(
    transforms=[
        PitchShift(16000, p=1, mode="per_batch"),
    ]
)

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Make an example tensor with white noise.
# This tensor represents 8 audio snippets with 2 channels (stereo) and 2 s of 16 kHz audio.
import time

audio_samples = (
    torch.rand(size=(8, 2, 32000), dtype=torch.float32, device=torch_device) - 0.5
)

# Apply augmentation. This varies the gain and polarity of (some of)
# the audio snippets in the batch independently.

start = time.process_time()
perturbed_audio_samples = apply_augmentation_per_example(audio_samples, sample_rate=16000)
print("per_example", time.process_time() - start)

start = time.process_time()
perturbed_audio_samples = apply_augmentation_per_channel(audio_samples, sample_rate=16000)
print("per_channel", time.process_time() - start)

start = time.process_time()
perturbed_audio_samples = apply_augmentation_per_batch(audio_samples, sample_rate=16000)
print("per_batch", time.process_time() - start)
