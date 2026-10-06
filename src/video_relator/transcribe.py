"""Speech-to-text for a video's narration, using faster-whisper (local, CPU)."""

import json
import os
from dataclasses import asdict, dataclass

# The Hugging Face cache warns about symlinks on Windows; it works fine without them.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from video_relator import ffmpeg  # noqa: E402
from video_relator.workspace import Workspace  # noqa: E402


@dataclass
class Segment:
    start: float
    end: float
    text: str


def _clock(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes):02d}:{secs:05.2f}"


def transcribe(ws: Workspace, *, model_size: str = "small", language: str | None = None) -> dict:
    """Transcribe ws.video and write the transcript files into its workspace.

    language=None lets Whisper detect it. Returns the same data written to transcript.json.
    """
    from faster_whisper import WhisperModel

    ws.ensure()
    if not ws.audio.exists():
        ffmpeg.extract_audio(ws.video, ws.audio)

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    # vad_filter skips stretches that are only music / silence
    raw_segments, info = model.transcribe(
        str(ws.audio), language=language, vad_filter=True, beam_size=5
    )
    segments = [
        Segment(round(s.start, 2), round(s.end, 2), s.text.strip()) for s in raw_segments
    ]

    result = {
        "source": ws.video.name,
        "model": model_size,
        "language": info.language,
        "language_probability": round(info.language_probability, 2),
        "duration": round(info.duration, 2),
        "segments": [asdict(s) for s in segments],
    }
    ws.transcript_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    ws.transcript_txt.write_text("\n".join(s.text for s in segments) + "\n", encoding="utf-8")
    ws.transcript_timed.write_text(
        "\n".join(f"[{_clock(s.start)} -> {_clock(s.end)}] {s.text}" for s in segments) + "\n",
        encoding="utf-8",
    )
    return result
