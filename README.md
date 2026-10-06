# video-relator

Transcribe the narration of a video, then re-narrate it with a synthetic voice and background music.
Works with any video; the voice follows the language of the original narration.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.14 and all dependencies).

```
uv sync
```

ffmpeg is used if it is on your `PATH`; otherwise the copy bundled with `imageio-ffmpeg` is used,
so nothing else needs installing. The speech models (Whisper, and Kokoro in `models/`) download
themselves the first time they are needed (a few hundred MB each); after that everything runs offline.

## Workflow

Put your videos in `media/` (git-ignored). For each video:

```
# 1. Transcribe the original narration
uv run video-relator transcribe media/my_video.mp4

# 2. Write the narration script (see below), then audition a voice on one block
uv run video-relator narrate media/my_video.mp4 --sample 1 --voice es_ES-davefx

# 3. Speak the whole script, create the music, and build the videos
uv run video-relator narrate media/my_video.mp4
uv run video-relator music media/my_video.mp4 --mood warm
uv run video-relator render media/my_video.mp4
```

`render` writes two videos: `<name>_narrated.mp4` (the original video stream untouched, new audio)
and `<name>_narrated_h264.mp4` (re-encoded H.264: much smaller, plays practically everywhere).
Use `--formats original|h264|both`, `--audio-only` to just check the soundtrack, and `--preview 20`
to render only the first 20 seconds.

| Command      | Options                                                                 |
|--------------|-------------------------------------------------------------------------|
| `transcribe` | `--model` Whisper size (`tiny`...`large-v3`, default `small`); `--language` (default: auto); `--force` |
| `narrate`    | `--script` (default `output/<name>/narration.txt`); `--voice` (default by language); `--sample N` |
| `music`      | `--mood` see below; `--seed`; `--file` to use your own audio file instead |
| `render`     | `--formats`, `--audio-only`, `--preview SECONDS`                        |
| all          | `--output` root folder for results (default `output`)                   |

## The narration script

Speech recognition is not perfect, so the transcript is only the starting point. Copy the lines of
`output/<name>/transcript.txt` into a script and correct them. The voice reads exactly what is written.

```
[1]
First paragraph to be spoken.

[2]
Second paragraph. [pause 1.2] A long silence before this sentence.
```

- **One block per transcript segment** (same count, same order), so each block lands where the original narrator spoke.
- **Punctuation drives the rhythm**: a full stop pauses longer than a semicolon or colon, and both longer than a comma. Write it for the listener.
- **`[pause 0.4]`** forces a silence of at least that many seconds at that spot. Put it at a clause boundary, right after a comma or semicolon.
- **`*a whole clause,*`** is spoken slower and a little louder, with a pause before and after. Emphasize complete clauses (punctuation inside the asterisks), typically a key figure or result.
- **`[?]`** marks a word you doubt; it is not spoken, `narrate` just warns about it.

`narrate` places each stretch of speech at the moment the original narrator started (found with a speech
detector). The voice may run on after the original narrator stopped, but never into the next picture change
(found once per video and cached in `cuts.json`), so pauses and emphasis do not force it to hurry. Only if a
scene still does not fit is the voice sped up (at most x1.2). It prints the fit of every scene.

## Voices

Two offline engines; the default voice follows the language of the original narration.

| `--voice`                                          | Engine | Accent / notes                                      |
|----------------------------------------------------|--------|-----------------------------------------------------|
| `es_ES-sharvard-F-smooth` (default for Spanish)    | Piper  | Spain (Castilian), female, steadier intonation. Dataset CC BY 3.0: **credit the authors** (Sharvard corpus, University of Edinburgh) |
| `es_ES-sharvard-F`, `es_ES-sharvard-M`             | Piper  | same corpus, default intonation / male voice (also has `-smooth`) |
| `es_ES-davefx`, `es_ES-davefx-smooth`              | Piper  | Spain (Castilian), male. Dataset CC0 (public domain, nothing to credit) |
| `ef_dora`, `em_alex`                               | Kokoro | Latin American Spanish                               |
| `af_heart`, `ff_siwis`, `if_sara`, `pf_dora`, ...  | Kokoro | English, French, Italian, Portuguese, Hindi, Japanese, Chinese |

Audition with `narrate --sample N --voice <name>` (writes `output/<name>/samples/`). An unknown name lists them all.
The Piper engine is GPL-3.0; that only matters if you redistribute this tool, not for the audio it produces.

## Cloned voices

`voice_clone/` is a separate tool (own environment: Python 3.12 + CUDA PyTorch) that clones the voices
of a recording with [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Apache 2.0) and speaks any text
with them, offline on the GPU. Run it from the repository root:

```
# 1. Find the speakers (named A, B... in order of appearance) and cut reference clips
uv run --project voice_clone voice-clone prepare media/recording.mp3 --speakers 2

# 2. Store a voice from one clip and exactly what it says
uv run --project voice_clone voice-clone save A --clip output/recording/voice_candidates/A1_213s.wav --text "..."

# 3. Speak a text, or a dialogue tagged [A] / [B]
uv run --project voice_clone voice-clone speak --file dialogue.txt --out speech.wav --takes 3
uv run --project voice_clone voice-clone speak --voice B --text "Hola a todos." --out hola.wav
uv run --project voice_clone voice-clone list
```

- `prepare` also repairs the recording: an inverted stereo channel (which cancels the voice in a mono
  mix) and hard clipping are fixed before anything else.
- The **reference clip matters**: 8-15 s, one voice, no music, and `--text` must match the audio word for
  word. Try a few candidates and keep the one that sounds closest.
- `--takes 3` generates every sentence group three times and keeps the take closest to the real voice
  that says every word. Slow on a GTX 1070: about 8x real time per take with the 1.7B model
  (`save --model 0.6b` is about twice as fast and a little less faithful).
### Dubbing a recording into another language

```
# 1. Timed script of the original: one block per stretch of speech, with its speaker
uv run --project voice_clone voice-clone transcribe media/recording.mp3 --speakers 2
#    -> output/recording/dub/script_es.txt

# 2. Translate every block into script_en.txt, keeping each "[n] A mm:ss.ss-mm:ss.ss" header line

# 3. Speak it with the speakers' own voices (saved as A, B...), each block at its original time
uv run --project voice_clone voice-clone dub media/recording.mp3 --script output/recording/dub/script_en.txt
#    -> output/recording/dub/recording_en.wav (+ .json: placement and quality of every block)
```

- Blocks that break mid-sentence can be merged in the translation: keep the first header and set its end
  time to the last block's end.
- Keep each block about as long as the original (roughly 2.5 words per second of its slot). A block that
  runs over is sped up by at most 15%; anything left pushes the next block a little later.
- The voices speak the new language with the original speaker's timbre and a light accent of their own
  language.

- Voices live in `voices/<name>/` (git-ignored: they are recordings of real people). Only clone
  voices you have permission to use.

## Music

Generated from scratch (no samples, so no copyright), as long as the video. Pick the mood that fits the topic:

| `--mood`    | Fits                                                          |
|-------------|---------------------------------------------------------------|
| `calm`      | neutral, works for anything (default)                         |
| `warm`      | food, agriculture, nature, people, culture                    |
| `serious`   | science, technical reports, finance, legal                    |
| `uplifting` | launches, results, achievements, motivation                   |

The music dips under the voice and swells in the pauses. To use a recorded track instead: `music --file track.mp3`.

## Layout

```
media/                     input videos (git-ignored)
models/                    downloaded voice models (git-ignored)
output/<video-name>/       everything generated for one video (contents git-ignored)
    transcript.json/.txt     the recognised narration (+ transcript_timed.txt)
    narration.txt            your narration script
    voice.wav                new narration, placed on a full-length track
    voice_timeline.json      where each block starts and ends
    music.wav                background music
    soundtrack.wav           final mix
    <name>_narrated*.mp4     the resulting videos
    samples/                 voice auditions
src/video_relator/
    cli.py                   command line entry point
    transcribe.py            speech-to-text (faster-whisper)
    script.py, timing.py     script parsing; when the original narrator speaks
    tts.py, narrate.py       voices (Kokoro, Piper) and placing them on the timeline
    music.py, mix.py         generated music; mixing with ducking
    render.py, ffmpeg.py     soundtrack and video output
voices/<name>/             cloned voices: reference clip + text, model prompt, speaker profile (git-ignored)
voice_clone/src/voice_clone/
    cli.py                   `voice-clone` entry point (prepare, save, speak, list, transcribe, dub)
    dub.py                   timed transcript by speaker; dubbed track at the original times
    recording.py             decode, fix inverted stereo, repair clipping
    prepare.py               who speaks when; reference clips per speaker
    voices.py, speak.py      stored voices; speaking text and [A]/[B] dialogues
    models.py                Qwen3-TTS, ECAPA speaker embeddings, Whisper
.claude/skills/narration-style/   house style for scripts, pauses and music (for Claude Code)
```
