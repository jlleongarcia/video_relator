"""Minimal 16-bit WAV reading/writing on numpy arrays (float, -1..1)."""

import wave
from pathlib import Path

import numpy as np


def write_wav(path: Path, data: np.ndarray, sample_rate: int) -> None:
    """Write mono (n,) or multichannel (n, channels) float audio as 16-bit PCM."""
    pcm = (np.clip(data, -1.0, 1.0) * 32767).astype("<i2")
    channels = 1 if pcm.ndim == 1 else pcm.shape[1]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV as float32: (n,) if mono, else (n, channels)."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit WAV is supported")
        channels, rate = w.getnchannels(), w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    data = pcm.astype(np.float32) / 32768.0
    return (data.reshape(-1, channels) if channels > 1 else data), rate
