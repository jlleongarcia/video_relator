"""Load a recording as clean mono: fix stereo phase inversion and repair hard clipping."""

import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from scipy.interpolate import CubicSpline

RATE = 48000
_CLIP_LEVEL = 0.94  # samples this loud are treated as clipped
_CONTEXT = 12  # good samples each side used to rebuild a clipped run
_MAX_OVERSHOOT = 1.6  # a rebuilt peak may reach at most this many times the clip level


def _decode(path: Path) -> np.ndarray:
    """Any audio/video file -> float32 array (samples, channels) at RATE."""
    raw = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path), "-vn",
         "-f", "f32le", "-acodec", "pcm_f32le", "-ar", str(RATE), "-"],
        capture_output=True, check=True).stdout
    probe = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
                           capture_output=True, text=True).stderr
    channels = 2 if ("stereo" in probe or "2 channels" in probe) else 1
    return np.frombuffer(raw, np.float32).reshape(-1, channels)


def _declip(x: np.ndarray) -> tuple[np.ndarray, int]:
    """Rebuild clipped runs with a cubic spline through the good samples around them."""
    x = x.astype(np.float64)
    bad = np.abs(x) >= _CLIP_LEVEL
    bad[1:] |= bad[:-1]  # one sample of margin each side: codec ringing at the edges
    bad[:-1] |= bad[1:]
    edges = np.flatnonzero(np.diff(np.r_[0, bad.astype(np.int8), 0]))
    runs = 0
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a > 200 or a < _CONTEXT or b + _CONTEXT > len(x):
            continue
        good = np.r_[np.arange(a - _CONTEXT, a), np.arange(b, b + _CONTEXT)]
        good = good[~bad[good]]
        if len(good) < 6:
            continue
        gap = np.arange(a, b)
        fit = CubicSpline(good, x[good])(gap)
        sign = np.sign(x[gap])
        fit = np.minimum(np.abs(fit) * (np.sign(fit) == sign), _MAX_OVERSHOOT * _CLIP_LEVEL)
        x[gap] = sign * np.maximum(fit, np.abs(x[gap]))  # never below the clipped value
        runs += 1
    return x, runs


def load_clean(path: Path) -> tuple[np.ndarray, dict]:
    """Mono float32 at RATE, plus a report of what was repaired."""
    audio = _decode(path)
    report = {"channels": audio.shape[1], "phase_inverted": False, "clipped_runs": 0}
    channels = []
    for c in range(audio.shape[1]):
        fixed, runs = _declip(audio[:, c])
        channels.append(fixed)
        report["clipped_runs"] += runs
    if len(channels) == 2:
        left, right = channels
        report["channel_correlation"] = round(float(np.corrcoef(left, right)[0, 1]), 3)
        # Strongly negative correlation: one channel is inverted, and a plain mix cancels the voice.
        report["phase_inverted"] = report["channel_correlation"] < -0.5
        mono = (left - right) / 2 if report["phase_inverted"] else (left + right) / 2
    else:
        mono = channels[0]
    peak = np.abs(mono).max()
    if peak > 0.99:
        mono *= 0.99 / peak
    return mono.astype(np.float32), report
