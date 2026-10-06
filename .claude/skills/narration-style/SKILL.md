---
name: narration-style
description: House style for preparing a video's narration script and choosing its background music in this repo. Use whenever writing or revising a narration script (output/<video>/narration.txt) from a transcript, or picking the music for a video.
---

# Narration style

Standing preferences of the repo owner for every video processed with `video-relator`.
They shape the *script* and the *soundtrack*; they never change what the narration says.

## 1. Enhance punctuation for better understanding

Speech recognition gives weak punctuation, and the synthetic voice reads exactly what is written,
so punctuation is the only control over intonation and rhythm. When turning `transcript.txt`
into the narration script, fix the punctuation together with the recognition errors:

- Break run-on sentences at their natural joints; prefer several clear sentences to one long one.
- Commas around insertions and between items of an enumeration; a colon before a list or an explanation;
  a full stop where the idea ends.
- Keep the wording and meaning intact. Change only punctuation and evident recognition errors.
- Check how numbers, percentages, units and acronyms are pronounced (`Narrator` uses espeak phonemes;
  e.g. "HPLC" is spelled out correctly in Spanish). If something is read wrongly, respell it in the script.
- The user reviews the script before any voice is added. Never change the words of a script they already
  approved. Delivery markup (punctuation, pauses, emphasis) is added only when they ask for it: keep a `.bak`
  copy first and check that the words are identical afterwards.

## 2. Be creative about pauses

The rhythm comes from punctuation (a full stop pauses longer than a semicolon or colon, both longer than a
comma, `...` longest). On top of that, shape the delivery with two kinds of markup, without changing any word:

- `[pause 0.4]` where a listener needs time to take something in: after a clause that announces something,
  between the steps of an enumeration (put it after the semicolon), before a contrast. Place it at a clause
  boundary (right after a comma or semicolon), never mid-clause. 0.3 to 0.6 s is usually enough; longer only
  at a real change of subject.
- `*a whole clause,*` to emphasize: spoken slower and a little louder with pauses around it. Reserve it for what
  the audience must remember, typically the key figures and results. Emphasize complete clauses with their
  punctuation inside the asterisks, and do not emphasize everything, or nothing stands out.
- Mind the fit: `narrate` reports each scene against the room available (until the next picture change). Extra
  pauses and emphasis add time; if a scene ends up sped up or runs into the next picture, remove some.

## 3. Match the music to the topic

Before generating music, read the transcript, decide the topic and the tone, and choose
`video-relator music ... --mood <mood>`:

| mood      | fits                                                              |
|-----------|-------------------------------------------------------------------|
| calm      | neutral, works for anything                                       |
| warm      | food, agriculture, nature, people, culture                        |
| serious   | science, technical reports, finance, legal                        |
| uplifting | launches, results, achievements, motivation                       |

Tell the user which mood you chose and why, and mention the alternatives. You cannot listen to audio,
so the user judges the result; `--mood` and `--seed` make it cheap to try another. Music is always generated
from scratch (or a file the user provides with `--file`): never fetch copyrighted music.

## Order of work

`transcribe` -> draft the script (punctuation, pauses) -> user reviews -> `narrate --sample N` to audition
a voice -> `narrate` -> `music --mood ...` -> `render --audio-only` to check -> `render`.
