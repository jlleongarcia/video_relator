"""Speak a text, or a dialogue between stored voices.

A dialogue script names the voice in brackets at the start of a turn; untagged text uses the
default voice. A blank line is a paragraph break (a longer pause).

    [A] Buenos días a todos, y bienvenidos.
    [B] Gracias. Vamos a empezar con el método.

With --takes N each sentence group is generated N times and the take whose voice is closest to the
speaker (and that says all the words) is kept.
"""

import re
from dataclasses import dataclass

import librosa
import numpy as np
import torch

from voice_clone import models
from voice_clone.voices import VOICES_DIR, Voice, load

_LANGUAGE_NAME = {"es": "Spanish", "en": "English", "fr": "French", "de": "German", "it": "Italian",
                  "pt": "Portuguese", "ru": "Russian", "ja": "Japanese", "ko": "Korean", "zh": "Chinese"}
_TAG = re.compile(r"^\s*\[([^\]]+)\]\s*")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_MAX_CHARS = 220  # longer pieces drift; split at sentence ends
_PAUSE = {"sentence": 0.35, "paragraph": 0.8, "turn": 0.6}
_TARGET_RMS = 0.08
_MAX_WORD_ERROR = 0.25  # a take that drops or invents more words than this is rejected


@dataclass
class Piece:
    voice: str
    text: str
    pause_after: float


def parse(script: str, default_voice: str | None) -> list[Piece]:
    pieces: list[Piece] = []
    voice = default_voice
    for paragraph in re.split(r"\n\s*\n", script.strip()):
        for line in paragraph.splitlines():
            if not line.strip():
                continue
            tag = _TAG.match(line)
            if tag:
                if pieces:
                    pieces[-1].pause_after = max(pieces[-1].pause_after, _PAUSE["turn"])
                voice, line = tag.group(1).strip(), line[tag.end():]
            if voice is None:
                raise ValueError("no voice: start the text with [Name] or pass --voice")
            group = ""
            for sentence in _SENTENCE_END.split(line.strip()):
                if group and len(group) + len(sentence) > _MAX_CHARS:
                    pieces.append(Piece(voice, group, _PAUSE["sentence"]))
                    group = ""
                group = f"{group} {sentence}".strip()
            if group:
                pieces.append(Piece(voice, group, _PAUSE["sentence"]))
        if pieces:
            pieces[-1].pause_after = max(pieces[-1].pause_after, _PAUSE["paragraph"])
    return pieces


def _words(s: str) -> list[str]:
    return re.sub(r"[^\w ]", " ", s.lower()).split()


def _word_error(ref: str, hyp: str) -> float:
    r, h = _words(ref), _words(hyp)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))


def _level(x: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2))
    return x * (_TARGET_RMS / rms) if rms > 0 else x


def _speak_piece(voice: Voice, text: str, takes: int, seed: int, check: bool, *, language: str | None = None,
                 good_enough: float | None = None) -> tuple[np.ndarray, int, dict]:
    """Up to `takes` takes; stops early at a complete take at least `good_enough` alike."""
    language = language or voice.language
    tts = models.tts(voice.model_size)
    best, best_key, info = None, None, {}
    for take in range(takes):
        torch.manual_seed(seed + take)
        wavs, rate = tts.generate_voice_clone(text=text, language=_LANGUAGE_NAME.get(language, "Auto"),
                                              voice_clone_prompt=voice.prompt)
        x = wavs[0].astype(np.float32)
        if takes == 1 and not check:
            return x, rate, {}
        x16 = librosa.resample(x, orig_sr=rate, target_sr=16000)
        similarity = float(models.speaker_embedding(x16) @ voice.profile)
        error = _word_error(text, models.transcribe(x16, language, size="small")) if check else 0.0
        key = (error <= _MAX_WORD_ERROR, similarity - error)  # complete takes first, then most alike
        if best_key is None or key > best_key:
            best, best_key = x, key
            info = {"take": take + 1, "similarity": round(similarity, 3), "word_error": round(error, 2)}
        if good_enough is not None and key[0] and similarity >= good_enough:
            break
    return best, rate, info


_voices: dict[str, Voice] = {}

# Stock (not cloned) Kokoro voices, named <language letter><gender letter>_<name>, e.g. af_heart, bf_emma:
# for minor speakers whose own voice is not needed. The letters give espeak's phonemizer language.
_KOKORO_VOICE = re.compile(r"^([abefhijpz][fm])_\w+$")
_KOKORO_LANGUAGE = {"af": "en-us", "am": "en-us", "bf": "en-gb", "bm": "en-gb", "ef": "es", "em": "es",
                    "ff": "fr-fr", "if": "it", "im": "it", "pf": "pt-br", "pm": "pt-br"}


def _speak_kokoro(text: str, voice: str) -> tuple[np.ndarray, int, dict]:
    tts = models.kokoro()
    if voice not in tts.get_voices():
        raise KeyError(f"unknown Kokoro voice '{voice}' (available: {', '.join(sorted(tts.get_voices()))})")
    out, rate = [], 24000
    for piece in parse(text, voice):
        x, rate = tts.create(piece.text, voice=voice, lang=_KOKORO_LANGUAGE[voice[:2]])
        out += [_level(x.astype(np.float32)), np.zeros(int(_PAUSE["sentence"] * rate), np.float32)]
    audio = np.concatenate(out[:-1])
    peak = np.abs(audio).max()
    return (audio * 0.99 / peak if peak > 0.99 else audio), rate, {}


def speak_block(text: str, voice: str, *, language: str, takes: int, seed: int,
                good_enough: float | None = None) -> tuple[np.ndarray, int, dict]:
    """One block of a dub: sentence groups, each its best take, joined with short pauses.

    `voice` is a stored (cloned) voice, or a stock Kokoro voice such as af_heart (spoken once, no takes).
    """
    if _KOKORO_VOICE.match(voice) and not (VOICES_DIR / voice).exists():
        return _speak_kokoro(text, voice)
    if voice not in _voices:
        _voices[voice] = load(voice)
    out, infos, rate = [], [], 24000
    for n, piece in enumerate(parse(text, voice)):
        x, rate, info = _speak_piece(_voices[voice], piece.text, takes, seed + n, takes > 1,
                                     language=language, good_enough=good_enough)
        out += [_level(x), np.zeros(int(_PAUSE["sentence"] * rate), np.float32)]
        infos.append(info)
    audio = np.concatenate(out[:-1])
    peak = np.abs(audio).max()
    if peak > 0.99:
        audio *= 0.99 / peak
    if not infos[0]:
        return audio, rate, {}
    return audio, rate, {"similarity": round(float(np.mean([i["similarity"] for i in infos])), 3),
                         "word_error": round(float(np.mean([i["word_error"] for i in infos])), 2),
                         "takes": [i["take"] for i in infos]}


def speak(script: str, *, default_voice: str | None, takes: int = 1, seed: int = 0,
          check: bool | None = None) -> tuple[np.ndarray, int, list[dict]]:
    """Returns (audio, sample rate, one log entry per piece)."""
    pieces = parse(script, default_voice)
    voices = {name: load(name) for name in dict.fromkeys(p.voice for p in pieces)}
    check = takes > 1 if check is None else check
    out, log, rate = [], [], 24000
    for n, piece in enumerate(pieces, 1):
        x, rate, info = _speak_piece(voices[piece.voice], piece.text, takes, seed + 1000 * n, check)
        out += [_level(x), np.zeros(int(piece.pause_after * rate), np.float32)]
        log.append({"voice": piece.voice, "text": piece.text, "seconds": round(len(x) / rate, 1)} | info)
        print(f"[{n}/{len(pieces)}] {piece.voice}: {piece.text[:70]}"
              + (f"  (take {info['take']}, similarity {info['similarity']})" if info else ""), flush=True)
    audio = np.concatenate(out[:-1]) if out else np.zeros(0, np.float32)
    peak = np.abs(audio).max(initial=0)
    if peak > 0.99:
        audio *= 0.99 / peak
    return audio, rate, log
