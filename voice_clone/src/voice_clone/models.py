"""The neural models, loaded on first use and kept for the rest of the run.

Everything is downloaded into models/ (not the Hugging Face cache, which needs symlinks on Windows).
"""

import os
from functools import cache
from pathlib import Path

import numpy as np
import torch

# ctranslate2 (Whisper) needs CUDA DLLs that PyTorch ships with; make Windows find them.
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))

MODELS_DIR = Path("models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TTS_MODELS = {"1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base"}


def _download(repo: str, folder: str) -> Path:
    path = MODELS_DIR / folder
    if not (path / "config.json").exists() and not (path / "hyperparams.yaml").exists():
        from huggingface_hub import snapshot_download

        print(f"Downloading {repo} to {path} (first time only)...")
        snapshot_download(repo, local_dir=path)
    return path


@cache
def _ecapa():
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    return EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", savedir=str(MODELS_DIR / "ecapa"),
                                          run_opts={"device": DEVICE}, local_strategy=LocalStrategy.COPY)


def speaker_embedding(audio16k: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """Unit-length ECAPA voice embedding(s) of 16 kHz audio; cosine similarity = dot product."""
    batch = [audio16k] if isinstance(audio16k, np.ndarray) else audio16k
    out = []
    for x in batch:  # one at a time: clips differ in length
        with torch.no_grad():
            e = _ecapa().encode_batch(torch.tensor(x, device=DEVICE)[None]).squeeze().cpu().numpy()
        out.append(e / np.linalg.norm(e))
    return out[0] if isinstance(audio16k, np.ndarray) else np.stack(out)


def speaker_embeddings_fixed(windows: np.ndarray) -> np.ndarray:
    """Embeddings of many equal-length 16 kHz windows (rows), batched."""
    out = []
    for i in range(0, len(windows), 64):
        with torch.no_grad():
            e = _ecapa().encode_batch(torch.tensor(windows[i:i + 64], device=DEVICE)).squeeze(1).cpu().numpy()
        out.append(e)
    e = np.concatenate(out)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


@cache
def whisper(size: str = "large-v3", device: str = DEVICE):
    from faster_whisper import WhisperModel

    return WhisperModel(size, device=device, compute_type="int8")


def transcribe(audio16k: np.ndarray, language: str, prompt: str | None = None, size: str = "large-v3") -> str:
    # The small model only checks takes: on the CPU it leaves all GPU memory to the speech model.
    model = whisper(size, "cpu") if size == "small" else whisper(size)
    segments, _ = model.transcribe(audio16k, language=language, beam_size=5,
                                   condition_on_previous_text=False, initial_prompt=prompt)
    return " ".join(s.text.strip() for s in segments).strip()


KOKORO_FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")  # shared with video-relator, which downloads them


@cache
def kokoro():
    from kokoro_onnx import Kokoro

    paths = [MODELS_DIR / name for name in KOKORO_FILES]
    if missing := [p for p in paths if not p.exists()]:
        raise ValueError(f"{', '.join(map(str, missing))} not found: run `video-relator narrate` with a Kokoro "
                         f"voice once to download them")
    return Kokoro(*map(str, paths))


@cache
def tts(size: str = "1.7b"):
    from qwen_tts import Qwen3TTSModel

    path = _download(TTS_MODELS[size], f"qwen3-tts-{size}-base")
    # float32: Pascal GPUs have no bfloat16 and very slow float16.
    return Qwen3TTSModel.from_pretrained(str(path), device_map=DEVICE, dtype=torch.float32,
                                         attn_implementation="sdpa")
