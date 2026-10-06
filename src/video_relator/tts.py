"""Text-to-speech, fully offline, with two engines:

- Kokoro (Apache 2.0): many languages, but its Spanish voices are Latin American.
- Piper: the Castilian (Spain) Spanish voices.

Pick a voice by name (--voice); `Narrator` hides which engine speaks it.
"""

import re
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from video_relator.errors import UserError

MODELS_DIR = Path("models")

# ---- Kokoro ----------------------------------------------------------------------------
_KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
_KOKORO_FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")
# Kokoro voices are named <language letter><gender letter>_<name>; the first two letters
# tell which espeak language to phonemize with.
_KOKORO_LANGUAGE = {
    "ef": "es", "em": "es", "af": "en-us", "am": "en-us", "bf": "en-gb", "bm": "en-gb",
    "ff": "fr-fr", "if": "it", "im": "it", "pf": "pt-br", "pm": "pt-br",
    "hf": "hi", "hm": "hi", "jf": "ja", "jm": "ja", "zf": "cmn", "zm": "cmn",
}

# ---- Piper -----------------------------------------------------------------------------
_PIPER_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


@dataclass(frozen=True)
class _PiperVoice:
    folder: str  # path inside the piper-voices repository
    model: str  # file name without extension
    speaker: int | None = None
    noise_scale: float | None = None  # lower = steadier, smoother intonation
    noise_w: float | None = None  # lower = more regular rhythm


_PIPER_BASE = {
    # Dataset CC0 (public domain).
    "es_ES-davefx": _PiperVoice("es/es_ES/davefx/medium", "es_ES-davefx-medium"),
    # Dataset CC BY 3.0: crediting the authors (Sharvard corpus, Univ. of Edinburgh) is required.
    "es_ES-sharvard-M": _PiperVoice("es/es_ES/sharvard/medium", "es_ES-sharvard-medium", speaker=0),
    "es_ES-sharvard-F": _PiperVoice("es/es_ES/sharvard/medium", "es_ES-sharvard-medium", speaker=1),
}
# Every voice also comes as "<name>-smooth": about a quarter less random variation in
# intonation and rhythm than the model's default (0.667 / 0.8).
PIPER_VOICES = {
    **_PIPER_BASE,
    **{f"{name}-smooth": replace(voice, noise_scale=0.5, noise_w=0.6) for name, voice in _PIPER_BASE.items()},
}

# Default voice for each transcript language code (as detected by Whisper).
DEFAULT_VOICE = {
    "es": "es_ES-sharvard-F-smooth", "en": "af_heart", "fr": "ff_siwis", "it": "if_sara",
    "pt": "pf_dora", "hi": "hf_alpha", "ja": "jf_alpha", "zh": "zf_xiaobei",
}

# ---- Rhythm ----------------------------------------------------------------------------
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_PAUSE_MARK = re.compile(r"\[pause\s+(\d+(?:\.\d+)?)\]", re.IGNORECASE)  # explicit pause, in seconds
# Abbreviations whose full stop does not end a sentence.
_ABBREVIATIONS = {"dr.", "dra.", "sr.", "sra.", "prof.", "fig.", "aprox.", "pág.", "mr.", "mrs.", "ms.", "st."}
# Silence after a sentence or clause, by how it ends (seconds): the punctuation shapes the rhythm.
_PAUSE_AFTER = {".": 0.55, "!": 0.6, "?": 0.6, "…": 0.8, ",": 0.2, ";": 0.3, ":": 0.3}
_DEFAULT_PAUSE = 0.3
# *Emphasis*: a clause spoken slower and a little louder, set off by pauses.
_EMPHASIS = re.compile(r"\*([^*]+)\*")
_EMPHASIS_SPEED = 0.9
_EMPHASIS_GAIN = 1.2
_EMPHASIS_LEAD = 0.35  # minimum silence before an emphasized clause
_EMPHASIS_TAIL = 0.4  # ... and after it
# Every spoken clause is levelled to this loudness (within +-6 dB) so none jumps out.
_TARGET_RMS = 0.07
_FADE = 0.01  # seconds; soft edges so joins between clauses never click


def _sentences(text: str) -> list[str]:
    """Split into sentences, keeping abbreviations like "Dr." attached to what follows."""
    sentences: list[str] = []
    for chunk in _SENTENCE_END.split(text.strip()):
        if sentences and sentences[-1].split()[-1].lower() in _ABBREVIATIONS:
            sentences[-1] += " " + chunk
        elif chunk:
            sentences.append(chunk)
    return sentences


def _pause_after(sentence: str) -> float:
    return 0.8 if sentence.endswith("...") else _PAUSE_AFTER.get(sentence[-1], _DEFAULT_PAUSE)


def _polish(audio: np.ndarray, rate: int, lowpass_hz: float | None) -> np.ndarray:
    """Make one spoken clause smoother: gentle high-frequency roll-off, even loudness, soft edges."""
    if lowpass_hz:  # 2nd-order roll-off, no phase change; tames harsh "s" and hiss
        spectrum = np.fft.rfft(audio)
        spectrum /= np.sqrt(1 + (np.fft.rfftfreq(len(audio), 1 / rate) / lowpass_hz) ** 4)
        audio = np.fft.irfft(spectrum, n=len(audio))
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    if rms > 0:
        audio = audio * np.clip(_TARGET_RMS / rms, 0.5, 2.0)
    edge = min(int(_FADE * rate), len(audio) // 2)
    ramp = np.linspace(0, 1, edge, dtype=np.float32)
    audio = audio.astype(np.float32)
    audio[:edge] *= ramp
    audio[len(audio) - edge:] *= ramp[::-1]
    return audio


def _download(url: str, dest: Path) -> None:
    if not dest.exists():
        print(f"Downloading {dest.name} (one time only)...", flush=True)
        MODELS_DIR.mkdir(exist_ok=True)
        urllib.request.urlretrieve(url, dest)


class Narrator:
    """Speaks text with one voice. `sample_rate` is the rate of the audio it returns."""

    def __init__(self, voice: str) -> None:
        self.voice = voice
        if voice in PIPER_VOICES:
            self._init_piper(PIPER_VOICES[voice])
        else:
            self._init_kokoro(voice)

    def _init_piper(self, preset: _PiperVoice) -> None:
        from piper import PiperVoice

        for ext in (".onnx", ".onnx.json"):
            _download(f"{_PIPER_URL}/{preset.folder}/{preset.model}{ext}", MODELS_DIR / f"{preset.model}{ext}")
        self._piper = PiperVoice.load(MODELS_DIR / f"{preset.model}.onnx")
        self._preset = preset
        self.sample_rate = self._piper.config.sample_rate
        self._lowpass_hz = 7500  # Piper voices carry some harsh hiss above this
        self._say = self._say_piper

    def _init_kokoro(self, voice: str) -> None:
        from kokoro_onnx import Kokoro

        for name in _KOKORO_FILES:
            _download(f"{_KOKORO_URL}/{name}", MODELS_DIR / name)
        self._kokoro = Kokoro(*(str(MODELS_DIR / name) for name in _KOKORO_FILES))
        known = sorted(self._kokoro.get_voices())
        if voice not in known:
            raise UserError(f"Unknown voice {voice!r}.\n  Castilian Spanish (Piper): {', '.join(PIPER_VOICES)}\n"
                            f"  Kokoro: {', '.join(known)}")
        self._language = _KOKORO_LANGUAGE[voice[:2]]
        self.sample_rate = 24000
        self._lowpass_hz = None
        self._say = self._say_kokoro

    def _say_kokoro(self, sentence: str, speed: float) -> np.ndarray:
        samples, _ = self._kokoro.create(sentence, voice=self.voice, speed=speed, lang=self._language)
        return samples.astype(np.float32)

    def _say_piper(self, sentence: str, speed: float) -> np.ndarray:
        from piper import SynthesisConfig

        p = self._preset
        config = self._piper.config
        chunks = self._piper.synthesize(sentence, SynthesisConfig(
            speaker_id=p.speaker,
            length_scale=config.length_scale / speed,
            noise_scale=p.noise_scale if p.noise_scale is not None else config.noise_scale,
            noise_w_scale=p.noise_w if p.noise_w is not None else config.noise_w_scale,
        ))
        return np.concatenate([c.audio_int16_array.astype(np.float32) / 32768.0 for c in chunks])

    def speak(self, text: str, speed: float = 1.0) -> np.ndarray:
        """Synthesize text clause by clause; returns mono float32 at `sample_rate`.

        Delivery is driven by the text itself:
        - the silence after a clause depends on how it ends (. ! ? ... ; : ,);
        - `[pause 1.5]` forces a silence of at least that many seconds at that spot;
        - `*a whole clause*` is spoken slower and a little louder, with pauses around it
          (emphasize complete clauses, punctuation inside the asterisks).
        """
        units: list[tuple[str, bool, float]] = []  # (clause, emphasized, forced pause before it)
        forced = 0.0
        # split() with one capture group alternates: text, marker, text, marker, ...
        for i, piece in enumerate(_PAUSE_MARK.split(text)):
            if i % 2:
                forced = max(forced, float(piece))
                continue
            for j, span in enumerate(_EMPHASIS.split(piece)):
                for clause in _sentences(span):
                    units.append((clause, bool(j % 2), forced))
                    forced = 0.0
        if not units:
            raise UserError("Nothing to speak in this block")

        # A clause cut off mid-sentence gets a trailing comma so its intonation stays open.
        clauses = [c if k == len(units) - 1 or c[-1] in _PAUSE_AFTER else c + ","
                   for k, (c, _, _) in enumerate(units)]

        parts: list[np.ndarray] = []
        for k, (_, emphasized, forced_pause) in enumerate(units):
            if k:
                gap = max(forced_pause, _pause_after(clauses[k - 1]))
                if emphasized:
                    gap = max(gap, _EMPHASIS_LEAD)
                if units[k - 1][1]:
                    gap = max(gap, _EMPHASIS_TAIL)
                parts.append(np.zeros(int(gap * self.sample_rate), dtype=np.float32))
            audio = _polish(self._say(clauses[k], speed * (_EMPHASIS_SPEED if emphasized else 1.0)),
                            self.sample_rate, self._lowpass_hz)
            parts.append(audio * (_EMPHASIS_GAIN if emphasized else 1.0))
        return np.concatenate(parts)
