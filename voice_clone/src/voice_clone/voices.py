"""Stored voices: voices/<name>/ holds everything needed to speak with a cloned voice.

    voice.json     what it is: language, reference text, where it came from, model size
    reference.wav  the reference clip (exactly what `ref_text` says)
    prompt.pt      the model's encoding of the reference (codec tokens + speaker embedding)
    profile.npy    the speaker's average voice embedding over the whole recording, used to pick
                   the take that sounds most like them
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch

from voice_clone import models

VOICES_DIR = Path("voices")


@dataclass
class Voice:
    name: str
    dir: Path
    meta: dict
    profile: np.ndarray
    prompt: list  # [VoiceClonePromptItem]

    @property
    def language(self) -> str:
        return self.meta["language"]

    @property
    def model_size(self) -> str:
        return self.meta["model"]


def save(name: str, clip: Path, text: str, *, language: str, profile: Path | None,
         model_size: str = "1.7b", source: dict | None = None) -> Path:
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem  # noqa: F401 (for torch.load later)

    d = VOICES_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(clip, d / "reference.wav")
    audio, rate = librosa.load(d / "reference.wav", sr=None)
    prompt = models.tts(model_size).create_voice_clone_prompt(ref_audio=(audio, rate), ref_text=text)
    torch.save([vars(item) for item in prompt], d / "prompt.pt")
    if profile is not None:
        shutil.copyfile(profile, d / "profile.npy")
    else:  # no whole-recording profile: fall back to the reference clip itself
        np.save(d / "profile.npy", models.speaker_embedding(librosa.resample(audio, orig_sr=rate, target_sr=16000)))
    meta = {"name": name, "language": language, "model": model_size, "ref_text": text,
            "ref_seconds": round(len(audio) / rate, 2), "source": source or {}}
    (d / "voice.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return d


def load(name: str) -> Voice:
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

    d = VOICES_DIR / name
    if not (d / "voice.json").exists():
        known = sorted(p.name for p in VOICES_DIR.glob("*") if (p / "voice.json").exists())
        raise KeyError(f"voice '{name}' not found in {VOICES_DIR}/ (available: {', '.join(known) or 'none'})")
    meta = json.loads((d / "voice.json").read_text(encoding="utf-8"))
    items = [VoiceClonePromptItem(**fields) for fields in torch.load(d / "prompt.pt", weights_only=False)]
    for item in items:
        item.ref_spk_embedding = item.ref_spk_embedding.to(models.DEVICE)
        if item.ref_code is not None:
            item.ref_code = item.ref_code.to(models.DEVICE)
    return Voice(name, d, meta, np.load(d / "profile.npy"), items)
