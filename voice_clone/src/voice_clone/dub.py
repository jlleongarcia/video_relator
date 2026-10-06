"""Dub a recording into another language with the cloned voices of its own speakers.

1. `transcribe` writes the original as a timed script: one block per stretch of speech between
   pauses, each with its speaker and start/end time.
2. A translator rewrites the text of every block (keeping the header lines) into a new script.
3. `dub` speaks each block with its speaker's voice and puts it where the original block started.

Script format, one block per paragraph:

    [12] A 01:03.44-01:08.52
    The text of the block.
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np

from voice_clone import models
from voice_clone.prepare import diarize
from voice_clone.recording import RATE, load_clean

_GAP_SPLIT = 0.7  # a pause this long always starts a new block
_SENTENCE_GAP = 0.25  # after a full stop, a pause this long starts a new block...
_SENTENCE_MIN = 4.0  # ...once the block is at least this long
_BLOCK_MAX = 20.0  # past this, split at the next comma or full stop
_MIN_GAP = 0.15  # silence kept between consecutive blocks
_MAX_SPEEDUP = 1.15  # a block may be sped up this much to fit its slot


@dataclass
class Block:
    id: int
    speaker: str
    start: float
    end: float
    text: str


def _clock(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


def _seconds(clock: str) -> float:
    m, s = clock.split(":")
    return int(m) * 60 + float(s)


def write_script(blocks: list[Block], path: Path) -> None:
    path.write_text("\n\n".join(f"[{b.id}] {b.speaker} {_clock(b.start)}-{_clock(b.end)}\n{b.text}" for b in blocks)
                    + "\n", encoding="utf-8")


_HEADER = re.compile(r"^\[(\d+)\]\s+(\S+)\s+(\d+:\d+(?:\.\d+)?)-(\d+:\d+(?:\.\d+)?)\s*$")


def read_script(path: Path) -> list[Block]:
    blocks: list[Block] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        header = _HEADER.match(line)
        if header:
            n, speaker, a, b = header.groups()
            blocks.append(Block(int(n), speaker, _seconds(a), _seconds(b), ""))
        elif line.strip():
            if not blocks:
                raise ValueError(f"{path}: text before the first [n] header")
            blocks[-1].text = f"{blocks[-1].text} {line.strip()}".strip()
    if empty := [b.id for b in blocks if not b.text]:
        raise ValueError(f"{path}: blocks without text: {empty}")
    return blocks


def transcribe(recording: Path, out_dir: Path, *, language: str, speakers: int, topic: str | None) -> list[Block]:
    out_dir.mkdir(parents=True, exist_ok=True)
    audio, report = load_clean(recording)
    x16 = librosa.resample(audio, orig_sr=RATE, target_sr=16000)
    print(f"Loaded {len(audio) / RATE / 60:.1f} min (phase inverted: {report['phase_inverted']}, "
          f"clipped runs repaired: {report['clipped_runs']}); transcribing...", flush=True)

    segments, _ = models.whisper("large-v3").transcribe(
        x16, language=language, beam_size=5, word_timestamps=True, vad_filter=True,
        initial_prompt=topic, condition_on_previous_text=False)
    words = [w for s in segments for w in s.words]
    print(f"{len(words)} words; finding the speakers...", flush=True)

    times, labels = diarize(x16, speakers)
    names = [chr(ord("A") + k) for k in range(speakers)]

    groups: list[list] = []
    for w in words:
        if groups:
            block, prev = groups[-1], groups[-1][-1]
            gap, length = w.start - prev.end, prev.end - block[0].start
            ends_sentence = prev.word.strip().endswith((".", "?", "!", "…"))
            if (gap >= _GAP_SPLIT or (ends_sentence and gap >= _SENTENCE_GAP and length >= _SENTENCE_MIN)
                    or (length >= _BLOCK_MAX and prev.word.strip().endswith((",", ";", ":", ".", "?", "!")))):
                groups.append([])
        else:
            groups.append([])
        groups[-1].append(w)

    blocks = []
    for n, g in enumerate(groups, 1):
        a, b = g[0].start, g[-1].end
        inside = (times + 0.75 >= a) & (times + 0.75 <= b)  # window centres inside the block
        if not inside.any():
            inside = np.abs(times + 0.75 - (a + b) / 2) == np.abs(times + 0.75 - (a + b) / 2).min()
        speaker = names[np.bincount(labels[inside], minlength=speakers).argmax()]
        blocks.append(Block(n, speaker, round(a, 2), round(b, 2), "".join(w.word for w in g).strip()))

    (out_dir / "source.json").write_text(json.dumps({"recording": str(recording), "duration": len(audio) / RATE,
                                                     "blocks": [asdict(b) for b in blocks]},
                                                    ensure_ascii=False, indent=1), encoding="utf-8")
    write_script(blocks, out_dir / f"script_{language}.txt")
    return blocks


def _stretch(x: np.ndarray, rate: int, factor: float) -> np.ndarray:
    """Speed up by `factor` without changing the pitch (ffmpeg atempo: clean for small changes)."""
    import subprocess

    import imageio_ffmpeg

    raw = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-f", "f32le", "-ar", str(rate),
                          "-ac", "1", "-i", "-", "-filter:a", f"atempo={factor:.4f}", "-f", "f32le", "-"],
                         input=x.astype(np.float32).tobytes(), capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def dub(script: Path, out: Path, *, duration: float, language: str, takes: int, seed: int = 0,
        good_enough: float | None = None) -> list[dict]:
    """Speak every block of `script` and place it on a track as long as the original.

    Every spoken block is kept in <out folder>/blocks_<language>/, so an interrupted run resumes where it
    stopped; a block is spoken again only if its speaker or text changed in the script.
    """
    import soundfile as sf

    from voice_clone.speak import speak_block

    blocks = read_script(script)
    cache = out.parent / f"blocks_{language}"
    cache.mkdir(parents=True, exist_ok=True)
    track = np.zeros(int(duration * RATE) + RATE * 5, np.float32)
    log, cursor = [], 0.0
    for i, b in enumerate(blocks):
        wav, meta = cache / f"{b.id}.wav", cache / f"{b.id}.json"
        saved = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() and wav.exists() else None
        if saved and saved["text"] == b.text and saved["speaker"] == b.speaker:
            x, info = sf.read(wav, dtype="float32")[0], saved["info"]
        else:
            x, rate, info = speak_block(b.text, b.speaker, language=language, takes=takes,
                                        seed=seed + 1000 * b.id, good_enough=good_enough)
            x = librosa.resample(x, orig_sr=rate, target_sr=RATE) if rate != RATE else x
            sf.write(wav, x, RATE, subtype="FLOAT")
            meta.write_text(json.dumps({"speaker": b.speaker, "text": b.text, "info": info}, ensure_ascii=False),
                            encoding="utf-8")
        start = max(b.start, cursor)
        next_start = blocks[i + 1].start if i + 1 < len(blocks) else duration
        slot = max(next_start - _MIN_GAP - start, 0.5)
        seconds = len(x) / RATE
        factor = 1.0
        if seconds > slot:
            factor = min(seconds / slot, _MAX_SPEEDUP)
            x = _stretch(x, RATE, factor)
        a = int(start * RATE)
        track[a:a + len(x)] += x[:len(track) - a]
        cursor = start + len(x) / RATE + _MIN_GAP
        late = start - b.start
        entry = {"id": b.id, "speaker": b.speaker, "original": [b.start, b.end], "placed": round(start, 2),
                 "seconds": round(len(x) / RATE, 2), "speedup": round(factor, 3), "late": round(late, 2)} | info
        log.append(entry)
        print(f"[{b.id}/{blocks[-1].id}] {b.speaker} {_clock(start)}  {entry['seconds']:5.1f}s"
              + (f"  x{factor:.2f}" if factor > 1 else "") + (f"  {late:+.1f}s late" if late > 0.05 else "")
              + (f"  sim {info['similarity']}" if "similarity" in info else ""), flush=True)
        if i % 10 == 9 or i == len(blocks) - 1:  # a listenable track as we go
            end = int(max(duration, cursor) * RATE)
            sf.write(out, track[:end], RATE, subtype="PCM_16")
            out.with_suffix(".json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    return log
