import wave
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torchaudio

from torch_audiomentations.utils.io import Audio


@pytest.fixture
def stereo_wav(tmp_path):
    values = np.array(
        [[-32768, 32767], [-16384, 16384], [12345, -5432], [0, 100]],
        dtype=np.int16,
    )
    values = np.tile(values, (5, 1))
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(values.tobytes())
    return path, torch.from_numpy((values.T / 32768).astype(np.float32))


@pytest.fixture
def soundfile_backend(monkeypatch):
    sf = pytest.importorskip("soundfile")
    monkeypatch.delattr(torchaudio, "info", raising=False)

    def fail_torchaudio_load(*args, **kwargs):
        pytest.fail("The SoundFile path must not call torchaudio.load")

    monkeypatch.setattr(torchaudio, "load", fail_torchaudio_load)
    return sf


class TestAudio:
    @pytest.mark.parametrize(
        "sample_rate,sample_offset,num_samples,original_start,original_count",
        [
            (8000, 3, 7, 3, 7),
            (16000, 6, 10, 3, 5),
            (16000, 6, 5, 3, 2),
            (11025, 4, 4, 3, 3),
            (4000, 2, 7, 4, 14),
        ],
    )
    def test_file_metadata_and_exact_partial_read(
        self,
        stereo_wav,
        sample_rate,
        sample_offset,
        num_samples,
        original_start,
        original_count,
        monkeypatch,
    ):
        path, samples = stereo_wav
        audio = Audio(sample_rate=sample_rate, mono=False)
        if not hasattr(torchaudio, "info"):

            def fail_legacy_load(*args, **kwargs):
                pytest.fail("torchaudio 2.9 must use SoundFile for partial reads")

            monkeypatch.setattr(torchaudio, "load", fail_legacy_load)
        assert audio.get_audio_metadata(path) == (20, 8000)
        assert audio.get_num_samples(path) == int(np.ceil(20 * sample_rate / 8000))
        actual = audio(path, sample_offset=sample_offset, num_samples=num_samples)
        expected = samples[:, original_start : original_start + original_count]
        if sample_rate != 8000:
            expected = torchaudio.functional.resample(expected, 8000, sample_rate)
        expected = expected[:, :num_samples]
        expected = torch.nn.functional.pad(
            expected, (0, num_samples - expected.shape[-1])
        )

        assert actual.shape == (2, num_samples)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_soundfile_metadata_does_not_read_samples(
        self, stereo_wav, soundfile_backend, monkeypatch
    ):
        def fail_read(*args, **kwargs):
            pytest.fail("Metadata lookup must not decode audio samples")

        monkeypatch.setattr(soundfile_backend.SoundFile, "read", fail_read)
        path, _ = stereo_wav
        assert Audio.get_audio_metadata(path) == (20, 8000)

    @pytest.mark.parametrize("mono,channel", [(False, None), (True, None), (True, 2)])
    def test_soundfile_reads_only_requested_frames(
        self, stereo_wav, soundfile_backend, mono, channel, monkeypatch
    ):
        path, samples = stereo_wav
        calls = []
        original_read = soundfile_backend.SoundFile.read

        def track_read(audio_file, frames=-1, dtype="float64", always_2d=False):
            calls.append((audio_file.tell(), frames, dtype, always_2d))
            return original_read(audio_file, frames, dtype=dtype, always_2d=always_2d)

        monkeypatch.setattr(soundfile_backend.SoundFile, "read", track_read)
        actual = Audio(sample_rate=8000, mono=mono)(
            {"audio": path, "channel": channel}, sample_offset=3, num_samples=7
        )
        expected = samples[:, 3:10]
        if channel is not None:
            expected = expected[channel - 1 : channel]
        elif mono:
            expected = expected.mean(dim=0, keepdim=True)
        assert calls == [(3, 7, "float32", True)]
        assert actual.dtype == torch.float32
        assert actual.is_contiguous()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    @pytest.mark.parametrize("channels", [1, 2, 4])
    @pytest.mark.parametrize(
        "subtype", ["PCM_U8", "PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"]
    )
    def test_soundfile_wav_encodings_and_channel_layout(
        self, tmp_path, soundfile_backend, channels, subtype
    ):
        values = ((np.arange(256 * channels).reshape(256, channels) % 17) - 8) / 16
        path = tmp_path / "encoded.wav"
        soundfile_backend.write(path, values, 16000, subtype=subtype)
        assert Audio.get_audio_metadata(path) == (256, 16000)
        actual = Audio(sample_rate=16000, mono=False)(
            path, sample_offset=17, num_samples=31
        )
        expected = torch.from_numpy(values[17:48].T.astype(np.float32))
        assert actual.shape == (channels, 31)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    @pytest.mark.parametrize(
        "extension,format,subtype",
        [
            ("flac", "FLAC", "PCM_24"),
            ("ogg", "OGG", "VORBIS"),
            ("aiff", "AIFF", "PCM_16"),
            ("mp3", "MP3", "MPEG_LAYER_III"),
        ],
    )
    def test_soundfile_other_formats_partial_read(
        self, tmp_path, soundfile_backend, extension, format, subtype
    ):
        if not soundfile_backend.check_format(format, subtype):
            pytest.skip(
                "This libsndfile build does not support " + format + "/" + subtype
            )
        values = np.random.RandomState(42).uniform(-0.5, 0.5, (4096, 2))
        path = tmp_path / ("encoded." + extension)
        soundfile_backend.write(path, values, 16000, format=format, subtype=subtype)
        # For lossy codecs, compare the crop with the same file decoded in full.
        decoded, sample_rate = soundfile_backend.read(
            path, dtype="float32", always_2d=True
        )
        assert Audio.get_audio_metadata(path) == (len(decoded), sample_rate)
        actual = Audio(sample_rate=sample_rate, mono=False)(
            path, sample_offset=123, num_samples=789
        )
        assert actual.shape == (2, 789)
        torch.testing.assert_close(
            actual, torch.from_numpy(decoded[123:912].T.copy()), atol=1e-6, rtol=1e-6
        )

    @pytest.mark.parametrize(
        "sample_offset,num_samples", [(0, None), (3, None), (13, 7), (20, 0), (20, None)]
    )
    def test_soundfile_full_tail_and_empty_reads(
        self, stereo_wav, soundfile_backend, sample_offset, num_samples
    ):
        path, samples = stereo_wav
        actual = Audio(sample_rate=8000, mono=False)(
            str(path), sample_offset=sample_offset, num_samples=num_samples
        )
        stop = None if num_samples is None else sample_offset + num_samples
        torch.testing.assert_close(actual, samples[:, sample_offset:stop], atol=0, rtol=0)

    def test_soundfile_rejects_read_past_end(self, stereo_wav, soundfile_backend):
        path, _ = stereo_wav
        with pytest.raises(ValueError):
            Audio(sample_rate=8000)(path, sample_offset=14, num_samples=7)

    @pytest.mark.parametrize("tuple_metadata", [False, True])
    def test_legacy_metadata_and_load_are_preserved(self, monkeypatch, tuple_metadata):
        info = (
            (SimpleNamespace(length=20, rate=8000), None)
            if tuple_metadata
            else SimpleNamespace(num_frames=20, sample_rate=8000)
        )
        calls = []

        def legacy_info(path):
            calls.append(("info", path))
            return info

        samples = torch.arange(14, dtype=torch.float32).reshape(2, 7)

        def legacy_load(path, frame_offset, num_frames):
            calls.append(("load", path, frame_offset, num_frames))
            return samples, 8000

        monkeypatch.setattr(torchaudio, "info", legacy_info, raising=False)
        monkeypatch.setattr(torchaudio, "load", legacy_load)
        actual = Audio(sample_rate=8000, mono=False)(
            "legacy.wav", sample_offset=3, num_samples=7
        )
        assert calls == [("info", "legacy.wav"), ("load", "legacy.wav", 3, 7)]
        torch.testing.assert_close(actual, samples, atol=0, rtol=0)

    def test_tensor_input_resampling_keeps_gradient(self):
        samples = torch.linspace(-0.5, 0.5, 20).reshape(1, 20).requires_grad_()
        audio = Audio(sample_rate=16000, mono=False)
        actual = audio(
            {"samples": samples, "sample_rate": 8000},
            sample_offset=6,
            num_samples=10,
        )
        expected = torchaudio.functional.resample(samples[:, 3:8], 8000, 16000)

        torch.testing.assert_close(actual, expected)
        assert actual.shape == (1, 10)
        actual.sum().backward()
        assert samples.grad is not None
        assert torch.isfinite(samples.grad).all()
        assert torch.count_nonzero(samples.grad[:, 3:8]) > 0

    @pytest.mark.parametrize(
        "shape",
        [(1, 512), (1, 2, 512), (2, 1, 512)],
    )
    def test_rms_normalize(self, shape: tuple):
        samples = torch.rand(size=shape, dtype=torch.float32)
        normalized_samples = Audio.rms_normalize(samples)

        assert samples.shape == normalized_samples.shape

        normalized_rms = np.sqrt(np.mean(np.square(normalized_samples.numpy())))
        assert normalized_rms == pytest.approx(1.0)
