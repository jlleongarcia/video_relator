"""Find who speaks when in a recording and cut the best reference clips for each voice.

Speakers are named A, B, C... in order of first appearance. For each speaker the result is a set of
candidate clips (pause to pause, 6-16 s, one voice only, accurately transcribed), ranked by how
typical they are of that voice, plus the voice's profile: the mean embedding of all its speech.
"""

import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from faster_whisper.vad import VadOptions, get_speech_timestamps
from sklearn.cluster import AgglomerativeClustering, KMeans

from voice_clone import models
from voice_clone.recording import RATE, load_clean

_WIN, _HOP = 1.5, 0.5  # diarization windows (seconds)
_CLIP_MIN, _CLIP_TARGET, _CLIP_MAX = 6.0, 9.0, 16.0
_PAD = 0.12  # silence kept around a clip


def diarize(x16: np.ndarray, speakers: int) -> tuple[np.ndarray, np.ndarray]:
    """Speaker label of each speech window (times, labels), labels numbered by first appearance."""
    starts = np.arange(0, len(x16) / 16000 - _WIN, _HOP)
    windows = np.stack([x16[int(s * 16000):int(s * 16000) + int(_WIN * 16000)] for s in starts])
    rms = np.sqrt((windows ** 2).mean(1))
    keep = rms > np.percentile(rms, 25) * 1.5  # skip silence
    emb = models.speaker_embeddings_fixed(windows[keep])
    labels = AgglomerativeClustering(n_clusters=speakers, metric="cosine", linkage="average").fit_predict(emb)
    if np.bincount(labels).min() < 0.03 * len(labels):  # average linkage split off a few outliers
        labels = KMeans(speakers, n_init=10, random_state=0).fit_predict(emb)
    order = {old: new for new, old in enumerate(dict.fromkeys(labels))}
    return starts[keep], np.array([order[k] for k in labels])


def _chunks(x16: np.ndarray) -> list[tuple[float, float]]:
    """Stretches of speech of 6-16 s that start and end at a real pause."""
    speech = get_speech_timestamps(x16, VadOptions(min_silence_duration_ms=250, speech_pad_ms=60))
    chunks, current = [], []
    for region in speech:
        a, b = region["start"] / 16000, region["end"] / 16000
        if current and (a - current[-1][1] > 1.5 or b - current[0][0] > _CLIP_MAX):
            chunks.append(current)
            current = []
        current.append((a, b))
        if b - current[0][0] >= _CLIP_TARGET and len(current) > 1:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return [(c[0][0], c[-1][1]) for c in chunks if _CLIP_MIN <= c[-1][1] - c[0][0] <= _CLIP_MAX]


def prepare(recording: Path, out_dir: Path, *, speakers: int, language: str, per_speaker: int = 6,
            topic: str | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    audio, report = load_clean(recording)
    print(f"Loaded {len(audio) / RATE / 60:.1f} min; phase inverted: {report['phase_inverted']}; "
          f"clipped runs repaired: {report['clipped_runs']}")
    x16 = librosa.resample(audio, orig_sr=RATE, target_sr=16000)

    times, labels = diarize(x16, speakers)
    names = [chr(ord("A") + k) for k in range(speakers)]
    chunks = []
    for a, b in _chunks(x16):
        inside = (times >= a - 0.3) & (times + _WIN <= b + 0.3)
        if inside.sum() < 3:
            continue
        counts = np.bincount(labels[inside], minlength=speakers)
        if (counts > 0).sum() == 1:  # one voice only
            chunks.append({"speaker": names[counts.argmax()], "start": round(a, 2), "end": round(b, 2)})
    if not chunks:
        raise RuntimeError("no clean single-speaker stretches of speech found")

    emb = models.speaker_embedding([x16[int(c["start"] * 16000):int(c["end"] * 16000)] for c in chunks])
    result = {"recording": str(recording), "report": report, "speakers": {}}
    for name in names:
        mine = [i for i, c in enumerate(chunks) if c["speaker"] == name]
        if not mine:
            continue
        profile = emb[mine].mean(0)
        profile /= np.linalg.norm(profile)
        np.save(out_dir / f"profile_{name}.npy", profile)
        first = min(times[labels == names.index(name)])
        ranked = sorted(mine, key=lambda i: -float(emb[i] @ profile))[:per_speaker]
        clips = []
        for rank, i in enumerate(ranked, 1):
            c = chunks[i]
            clip = audio[int((c["start"] - _PAD) * RATE):int((c["end"] + _PAD) * RATE)]
            path = out_dir / f"{name}{rank}_{int(c['start'])}s.wav"
            sf.write(path, clip, RATE, subtype="PCM_16")
            text = models.transcribe(librosa.resample(clip, orig_sr=RATE, target_sr=16000), language, topic)
            clips.append(c | {"file": path.name, "typicality": round(float(emb[i] @ profile), 3), "text": text})
        result["speakers"][name] = {
            "first_heard": round(float(first), 1),
            "speech_seconds": round(sum(chunks[i]["end"] - chunks[i]["start"] for i in mine)),
            "clips": clips,
        }
    (out_dir / "candidates.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result
