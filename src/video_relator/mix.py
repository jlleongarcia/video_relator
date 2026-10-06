"""Mix narration and music: the music dips under the voice and swells back in the pauses."""

import numpy as np

from video_relator.errors import UserError

RATE = 48000
LEVEL_DB = -20.0  # voice and music are each brought to this loudness first
DUCK_DB = -18.0  # music level, relative to that, while someone speaks
OPEN_DB = -6.0  # music level in the pauses
_HOP = RATE // 100  # 10 ms analysis frames
_SPEECH_FLOOR_DB = -50.0


def _to_level(x: np.ndarray, *, speech_only: bool) -> np.ndarray:
    """Scale so the RMS loudness is LEVEL_DB (ignoring the silent parts when speech_only)."""
    measured = x[np.abs(x) > 1e-3] if speech_only else x
    if measured.size == 0:
        raise UserError("The narration track is silent")
    return x * (10 ** (LEVEL_DB / 20) / np.sqrt(np.mean(measured.astype(np.float64) ** 2)))


def _speech_presence(voice: np.ndarray) -> np.ndarray:
    """Per-sample value from 0 (nobody speaks) to 1 (speaking), with ~1 s smooth transitions."""
    frames = len(voice) // _HOP
    rms = np.sqrt(np.mean(voice[:frames * _HOP].reshape(frames, _HOP) ** 2, axis=1))
    active = (rms > 10 ** (_SPEECH_FLOOR_DB / 20)).astype(np.float32)
    held = (np.convolve(active, np.ones(61, dtype=np.float32), mode="same") > 0).astype(np.float32)  # bridge +-0.3 s
    ramp = np.hanning(101)
    smooth = np.convolve(held, ramp / ramp.sum(), mode="same")
    return np.interp(np.arange(len(voice)), (np.arange(frames) + 0.5) * _HOP, smooth)


def build(voice: np.ndarray, music: np.ndarray) -> np.ndarray:
    """voice: mono (n,), music: stereo (>= n, 2), both at RATE. Returns stereo (n, 2)."""
    n = len(voice)
    if len(music) < n:
        raise UserError(f"The music ({len(music) / RATE:.1f}s) is shorter than the narration ({n / RATE:.1f}s); "
                        "run the music command again")
    voice = _to_level(voice, speech_only=True)
    music = _to_level(music[:n], speech_only=False)

    music_db = OPEN_DB + (DUCK_DB - OPEN_DB) * _speech_presence(voice)
    return music * (10 ** (music_db / 20)).astype(np.float32)[:, None] + voice[:, None]
