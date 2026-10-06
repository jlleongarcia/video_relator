"""Calm ambient background music generated from scratch: no samples, so no copyright.

Slow pad chords + soft bell notes + reverb, for any length. The same seed gives the same music.
Pick the mood that fits the video's topic (see MOODS).
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from video_relator import ffmpeg
from video_relator.wavio import write_wav
from video_relator.workspace import Workspace

RATE = 48000
CHORD_TAIL = 4.0  # each chord fades out this long while the next one fades in


@dataclass(frozen=True)
class Mood:
    description: str
    progression: list[list[int]]  # chords as MIDI notes, looped
    chord_seconds: float
    bell_step: float  # seconds between possible bell notes
    bell_rest: float  # chance that a bell slot is left silent


MOODS = {
    "calm": Mood(
        "neutral, soft and open: a safe choice for any topic",
        [[48, 55, 59, 64], [45, 55, 60, 64], [41, 57, 60, 64], [43, 59, 62, 64]],  # Cmaj7 Am7 Fmaj7 G6
        8.0, 1.5, 0.35),
    "warm": Mood(
        "mellow and friendly, slower: food, agriculture, nature, people, culture",
        [[41, 48, 57, 64], [48, 55, 62, 64], [50, 57, 60, 65], [46, 53, 57, 62]],  # Fmaj7 Cadd9 Dm7 Bbmaj7
        9.0, 2.0, 0.4),
    "serious": Mood(
        "understated and focused, minor colour, sparse: science, reports, finance, legal",
        [[45, 55, 60, 64], [41, 57, 60, 64], [48, 55, 59, 64], [40, 55, 59, 62]],  # Am7 Fmaj7 Cmaj7 Em7
        8.0, 2.0, 0.5),
    "uplifting": Mood(
        "brighter and a little livelier: launches, results, achievements, motivation",
        [[48, 55, 60, 64], [43, 55, 59, 62], [45, 55, 60, 64], [41, 57, 60, 65]],  # C G Am F
        6.0, 1.0, 0.25),
}
DEFAULT_MOOD = "calm"


def _hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def _pad(n: int, rng: np.random.Generator, mood: Mood) -> np.ndarray:
    """Sustained chords: a slightly detuned pair of soft tones per note, one in each ear."""
    out = np.zeros((n, 2), dtype=np.float32)
    for c in range(math.ceil(n / RATE / mood.chord_seconds)):
        start = int(c * mood.chord_seconds * RATE)
        stop = min(n, start + int((mood.chord_seconds + CHORD_TAIL) * RATE))
        t = np.arange(stop - start) / RATE
        rise = 0.5 - 0.5 * np.cos(np.pi * np.clip(t / 2.5, 0, 1))
        fall = 0.5 + 0.5 * np.cos(np.pi * np.clip((t - mood.chord_seconds) / CHORD_TAIL, 0, 1))
        envelope = rise * fall
        for midi in mood.progression[c % len(mood.progression)]:
            for side, cents in ((0, -4.0), (1, 4.0)):
                f = _hz(midi) * 2 ** (cents / 1200)
                phase = rng.uniform(0, 2 * np.pi, 3)
                tone = (np.sin(2 * np.pi * f * t + phase[0])
                        + 0.35 * np.sin(4 * np.pi * f * t + phase[1])
                        + 0.12 * np.sin(6 * np.pi * f * t + phase[2]))
                out[start:stop, side] += (envelope * tone * 0.05).astype(np.float32)
    return out


def _bells(n: int, rng: np.random.Generator, mood: Mood) -> np.ndarray:
    """Sparse, soft plucked notes taken from the current chord."""
    out = np.zeros((n, 2), dtype=np.float32)
    length = 4 * RATE
    t = np.arange(length) / RATE
    body = (1 - np.exp(-t / 0.008)) * np.exp(-t / 1.1)
    overtone = 0.3 * np.exp(-t / 0.35)
    for k in range(int(n / RATE / mood.bell_step)):
        at = k * mood.bell_step + rng.uniform(0, 0.1)
        if rng.random() < mood.bell_rest:  # leave some silence between notes
            continue
        chord = mood.progression[int(at // mood.chord_seconds) % len(mood.progression)]
        f = _hz(int(rng.choice(chord[1:])) + 12)
        i = int(at * RATE)
        stop = min(n, i + length)
        if stop <= i:
            break
        tt = t[:stop - i]
        note = (np.sin(2 * np.pi * f * tt) + overtone[:stop - i] * np.sin(4 * np.pi * f * tt)) \
            * body[:stop - i] * rng.uniform(0.5, 1.0) * 0.05
        pan = rng.uniform(0.2, 0.8)
        out[i:stop, 0] += ((1 - pan) * note).astype(np.float32)
        out[i:stop, 1] += (pan * note).astype(np.float32)
    return out


def _reverb(x: np.ndarray, rng: np.random.Generator, rt60: float = 3.0, wet: float = 0.45) -> np.ndarray:
    """Convolve with a synthetic decaying-noise room (a different one per ear)."""
    ir_len = int(rt60 * RATE)
    decay = np.exp(-6.9 * (np.arange(ir_len) / RATE) / rt60)
    size = 1 << (len(x) + ir_len - 1).bit_length()
    out = np.zeros_like(x)
    for ch in range(2):
        ir = np.convolve(rng.standard_normal(ir_len) * decay, np.ones(8) / 8, mode="same")  # darker tail
        ir = (ir / np.sqrt(np.sum(ir ** 2))).astype(np.float32)
        tail = np.fft.irfft(np.fft.rfft(x[:, ch], size) * np.fft.rfft(ir, size), size)[:len(x)]
        out[:, ch] = (1 - wet) * x[:, ch] + wet * tail
    return out


def generate(seconds: float, seed: int = 7, mood: str = DEFAULT_MOOD) -> np.ndarray:
    """Stereo float32 music of the given length at RATE (peak-normalized, with fades)."""
    chosen = MOODS[mood]
    n = int(seconds * RATE)
    rng = np.random.default_rng(seed)
    music = _reverb(_pad(n, rng, chosen) + _bells(n, rng, chosen), rng)

    t = np.arange(n) / RATE
    music *= (1 + 0.08 * np.sin(2 * np.pi * t / 23.0)).astype(np.float32)[:, None]  # slow breathing

    fade_in, fade_out = int(min(4.0, seconds / 4) * RATE), int(min(8.0, seconds / 4) * RATE)
    music[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)[:, None]
    music[n - fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)[:, None]
    return music * (0.9 / np.abs(music).max())


def make(ws: Workspace, *, seed: int = 7, mood: str = DEFAULT_MOOD, file: Path | None = None) -> Path:
    """Write ws.music as long as the video: generated, or imported from an existing audio file."""
    ws.ensure()
    seconds = ffmpeg.duration(ws.video)
    if file:
        ffmpeg.import_music(file, ws.music, seconds)
    else:
        write_wav(ws.music, generate(seconds, seed, mood), RATE)
    return ws.music
