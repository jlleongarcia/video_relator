"""Build the new narration: the script, spoken and placed where the original narrator spoke."""

import json
import math
from pathlib import Path

import numpy as np

from video_relator import ffmpeg
from video_relator.errors import UserError
from video_relator.script import Block, load_script
from video_relator.timing import find_scenes
from video_relator.tts import DEFAULT_VOICE, Narrator
from video_relator.wavio import write_wav
from video_relator.workspace import Workspace

OUT_RATE = 48000  # the rate video soundtracks use
BLOCK_PAUSE = 0.7  # seconds of silence between two blocks in the same scene
CUT_MARGIN = 0.3  # the voice should finish this long before the next picture change
MAX_SPEED = 1.2  # never speed the voice up beyond this


def _picture_changes(ws: Workspace) -> list[float]:
    if not ws.cuts.exists():
        print("Scanning the video for picture changes (done once per video)...", flush=True)
        ws.cuts.write_text(json.dumps(ffmpeg.scene_changes(ws.video)), encoding="utf-8")
    return json.loads(ws.cuts.read_text(encoding="utf-8"))


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Change the sample rate by cutting or padding the spectrum (band-limited, no aliasing)."""
    if src == dst:
        return x
    length = round(len(x) * dst / src)
    return (np.fft.irfft(np.fft.rfft(x), n=length) * (length / len(x))).astype(np.float32)


def _load_transcript(ws: Workspace) -> dict:
    if not ws.transcript_json.exists():
        raise UserError(f"No transcript for {ws.video.name}. Run: video-relator transcribe {ws.video}")
    return json.loads(ws.transcript_json.read_text(encoding="utf-8"))


def _pick_voice(transcript: dict, voice: str | None) -> str:
    if voice:
        return voice
    language = transcript["language"]
    if language not in DEFAULT_VOICE:
        raise UserError(f"No default voice for language {language!r}; choose one with --voice")
    return DEFAULT_VOICE[language]


def _speak_scene(narrator: Narrator, blocks: list[Block], speed: float):
    """Speak consecutive blocks back to back; returns (audio, [(block number, start, end)])."""
    rate = narrator.sample_rate
    pause = np.zeros(int(BLOCK_PAUSE * rate), dtype=np.float32)
    parts: list[np.ndarray] = []
    spans: list[tuple[int, float, float]] = []
    cursor = 0
    for block in blocks:
        audio = narrator.speak(block.text, speed)
        if parts:
            parts.append(pause)
            cursor += len(pause)
        spans.append((block.number, cursor / rate, (cursor + len(audio)) / rate))
        parts.append(audio)
        cursor += len(audio)
    return np.concatenate(parts), spans


def sample(ws: Workspace, script_path: Path, number: int, voice: str | None = None) -> Path:
    """Speak one block on its own, to audition a voice."""
    blocks = {b.number: b for b in load_script(script_path)}
    if number not in blocks:
        raise UserError(f"{script_path} has no block [{number}]")
    voice = _pick_voice(_load_transcript(ws), voice)

    narrator = Narrator(voice)
    audio = narrator.speak(blocks[number].text)
    dest = ws.samples_dir / f"{voice}_block{number}.wav"
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_wav(dest, audio, narrator.sample_rate)
    print(f"{dest}  ({len(audio) / narrator.sample_rate:.1f} s)")
    return dest


def narrate(ws: Workspace, script_path: Path, voice: str | None = None) -> dict:
    """Speak the whole script and place it on a full-length track (voice.wav)."""
    blocks = load_script(script_path)
    transcript = _load_transcript(ws)
    segments = transcript["segments"]
    if len(blocks) != len(segments):
        raise UserError(
            f"{script_path} has {len(blocks)} blocks but the original narration has {len(segments)} "
            f"segments (see {ws.transcript_txt}). Keep one block per segment so each lands in the right place."
        )
    voice = _pick_voice(transcript, voice)
    for block in blocks:
        if block.doubtful:
            print(f"warning: block [{block.number}] still has a [?] mark; it is spoken as written")

    ws.ensure()
    total = ffmpeg.duration(ws.video)
    scenes = find_scenes(ws.audio, segments)
    cuts = _picture_changes(ws)
    narrator = Narrator(voice)

    placed: list[tuple[float, np.ndarray]] = []
    report = []
    for n, scene in enumerate(scenes):
        # The voice may run on after the original narrator stopped, but not into the next picture
        # (nor into the next scene); it is never asked to finish before the original did.
        next_cut = min((c for c in cuts if c > scene.end), default=total)
        next_scene = scenes[n + 1].start - 1.0 if n + 1 < len(scenes) else total
        latest_end = max(scene.end, min(next_cut - CUT_MARGIN, next_scene))
        scene_blocks = [blocks[i] for i in scene.blocks]

        speed = 1.0
        audio, spans = _speak_scene(narrator, scene_blocks, speed)
        duration = len(audio) / narrator.sample_rate
        if scene.start + duration > latest_end:  # too long for its slot: speak faster, once
            speed = min(MAX_SPEED, round(duration / (latest_end - scene.start) * 1.02, 2))
            audio, spans = _speak_scene(narrator, scene_blocks, speed)
            duration = len(audio) / narrator.sample_rate

        placed.append((scene.start, audio))
        report.append({
            "start": round(scene.start, 2),
            "end": round(scene.start + duration, 2),
            "original_end": round(scene.end, 2),
            "latest_end": round(latest_end, 2),
            "speed": speed,
            "blocks": [{"number": num, "start": round(scene.start + a, 2), "end": round(scene.start + b, 2)}
                       for num, a, b in spans],
        })

    length = int(math.ceil(max(total, max(r["end"] for r in report)) * OUT_RATE))
    track = np.zeros(length, dtype=np.float32)
    for start, audio in placed:
        up = _resample(audio, narrator.sample_rate, OUT_RATE)
        i = int(start * OUT_RATE)
        track[i:i + len(up)] += up
    write_wav(ws.voice, track, OUT_RATE)

    result = {"voice": voice, "language": transcript["language"], "scenes": report}
    ws.timeline.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Voice: {voice}")
    for i, r in enumerate(report, 1):
        blocks_txt = ",".join(str(b["number"]) for b in r["blocks"])
        note = "  <- runs into the next picture change" if r["end"] > r["latest_end"] + 0.05 else ""
        print(f"  scene {i} (blocks {blocks_txt}): {r['start']:6.1f}s -> {r['end']:6.1f}s "
              f"(original ended {r['original_end']:6.1f}s, room until {r['latest_end']:6.1f}s), "
              f"speed x{r['speed']}{note}")
    return result
