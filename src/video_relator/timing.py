"""Find when the original narrator speaks, so the new voice can land in the same places.

Whisper's own segment times are unreliable after long silences (a block's start gets glued
to the previous block's end). The speech detector run directly on the audio is accurate, so
scenes are its continuous stretches of speech, and each transcript segment is assigned to
the scene it overlaps the most.
"""

from dataclasses import dataclass
from pathlib import Path

from video_relator.errors import UserError

_RATE = 16000


@dataclass
class Scene:
    start: float
    end: float
    blocks: list[int]  # 0-based indexes of the transcript segments spoken in this scene


def find_scenes(audio: Path, segments: list[dict], *, min_silence_ms: int = 2000) -> list[Scene]:
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    chunks = get_speech_timestamps(
        decode_audio(str(audio), sampling_rate=_RATE),
        VadOptions(min_silence_duration_ms=min_silence_ms, speech_pad_ms=100, min_speech_duration_ms=500),
    )
    scenes = [Scene(c["start"] / _RATE, c["end"] / _RATE, []) for c in chunks]
    if not scenes:
        raise UserError(f"No speech detected in {audio}")

    for i, seg in enumerate(segments):
        overlaps = [max(0.0, min(seg["end"], s.end) - max(seg["start"], s.start)) for s in scenes]
        best = max(range(len(scenes)), key=overlaps.__getitem__)
        if overlaps[best] == 0:  # no overlap at all: take the closest scene
            mid = (seg["start"] + seg["end"]) / 2
            best = min(range(len(scenes)), key=lambda j: abs((scenes[j].start + scenes[j].end) / 2 - mid))
        scenes[best].blocks.append(i)
    return [s for s in scenes if s.blocks]
