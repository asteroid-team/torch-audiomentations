import torch
from torch_audiomentations import Compose, PitchShift


# Initialize augmentation callable
apply_augmentation = Compose(
    transforms=[
        PitchShift(16000, p=1),
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
perturbed_audio_samples = apply_augmentation(audio_samples, sample_rate=16000)
print(time.process_time() - start)
