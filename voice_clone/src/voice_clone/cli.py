"""Command line entry point: `voice-clone <command> ...` (run from the repository root)."""

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

OUTPUT_ROOT = Path("output")


class UserError(Exception):
    """A problem the user can fix; shown without a traceback."""


def _cmd_prepare(args: argparse.Namespace) -> None:
    from voice_clone.prepare import prepare

    if not args.recording.exists():
        raise UserError(f"{args.recording} not found")
    out_dir = OUTPUT_ROOT / args.recording.stem / "voice_candidates"
    result = prepare(args.recording, out_dir, speakers=args.speakers, language=args.language,
                     topic=args.topic)
    for name, s in result["speakers"].items():
        print(f"\nVoice {name}: first heard at {s['first_heard']:.0f}s, {s['speech_seconds']}s of clean speech")
        for c in s["clips"]:
            print(f"  {c['file']:16s} typicality {c['typicality']}  {c['text'][:90]}")
    print(f"\nCandidates in {out_dir}. Check a clip's text against the audio, then:\n"
          f"  voice-clone save <name> --clip {out_dir / '<file>'} --text \"...\"")


def _cmd_save(args: argparse.Namespace) -> None:
    from voice_clone import voices

    if not args.clip.exists():
        raise UserError(f"{args.clip} not found")
    candidates = args.clip.parent / "candidates.json"
    source, profile = {}, None
    if candidates.exists():  # a clip from `prepare`: keep its origin and the speaker's profile
        data = json.loads(candidates.read_text(encoding="utf-8"))
        for speaker, s in data["speakers"].items():
            for c in s["clips"]:
                if c["file"] == args.clip.name:
                    source = {"recording": data["recording"], "speaker": speaker,
                              "start": c["start"], "end": c["end"]}
                    profile = args.clip.parent / f"profile_{speaker}.npy"
    if args.profile:
        if not args.profile.exists():
            raise UserError(f"{args.profile} not found")
        profile = args.profile
        candidates = args.profile.parent / "candidates.json"
        if not source and candidates.exists():
            recording = json.loads(candidates.read_text(encoding="utf-8"))["recording"]
            source = {"recording": recording, "speaker": args.profile.stem.removeprefix("profile_")}
    path = voices.save(args.name, args.clip, args.text, language=args.language, profile=profile,
                       model_size=args.model, source=source)
    print(f"Voice '{args.name}' saved in {path}")


def _cmd_speak(args: argparse.Namespace) -> None:
    from voice_clone.speak import speak

    if (args.text is None) == (args.file is None):
        raise UserError("give either --text or --file")
    script = args.text if args.text is not None else args.file.read_text(encoding="utf-8")
    try:
        audio, rate, log = speak(script, default_voice=args.voice, takes=args.takes, seed=args.seed,
                                 check=args.check)
    except (KeyError, ValueError) as error:
        raise UserError(str(error).strip("'\"")) from None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, audio, rate)
    args.out.with_suffix(".json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(audio) / rate:.1f}s of speech written to {args.out}")


def _dub_dir(recording: Path) -> Path:
    return OUTPUT_ROOT / recording.stem / "dub"


def _cmd_transcribe(args: argparse.Namespace) -> None:
    from voice_clone.dub import transcribe

    if not args.recording.exists():
        raise UserError(f"{args.recording} not found")
    out_dir = _dub_dir(args.recording)
    blocks = transcribe(args.recording, out_dir, language=args.language, speakers=args.speakers, topic=args.topic)
    turns = sum(1 for a, b in zip(blocks, blocks[1:]) if a.speaker != b.speaker)
    print(f"{len(blocks)} blocks, {turns} changes of speaker, written to {out_dir / f'script_{args.language}.txt'}")


def _cmd_dub(args: argparse.Namespace) -> None:
    from voice_clone.dub import dub

    source = _dub_dir(args.recording) / "source.json"
    if not source.exists():
        raise UserError(f"{source} not found: run `voice-clone transcribe {args.recording}` first")
    if not args.script.exists():
        raise UserError(f"{args.script} not found")
    duration = json.loads(source.read_text(encoding="utf-8"))["duration"]
    out = args.out or _dub_dir(args.recording) / f"{args.recording.stem}_{args.language}.wav"
    try:
        log = dub(args.script, out, duration=duration, language=args.language, takes=args.takes,
                  seed=args.seed, good_enough=args.good_enough)
    except (KeyError, ValueError) as error:
        raise UserError(str(error).strip("'\"")) from None
    sped = [e for e in log if e["speedup"] > 1]
    late = [e for e in log if e["late"] > 0.5]
    print(f"Dub written to {out}: {len(log)} blocks, {len(sped)} sped up, {len(late)} more than 0.5 s late")


def _cmd_list(args: argparse.Namespace) -> None:
    from voice_clone.voices import VOICES_DIR

    for d in sorted(VOICES_DIR.glob("*/voice.json")):
        meta = json.loads(d.read_text(encoding="utf-8"))
        src = meta.get("source", {})
        origin = Path(src["recording"]).name if src else "?"
        if "start" in src:
            origin += f" {src['start']:.0f}-{src['end']:.0f}s"
        if "speaker" in src:
            origin += f" (speaker {src['speaker']})"
        print(f"{meta['name']:12s} {meta['language']}  model {meta['model']}  from {origin}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voice-clone",
                                     description="Clone the voices of a recording and speak text with them.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="find the speakers in a recording and cut reference clips")
    p.add_argument("recording", type=Path)
    p.add_argument("--speakers", type=int, default=2, help="how many different voices (default: 2)")
    p.add_argument("--language", default="es", help="language code (default: es)")
    p.add_argument("--topic", default=None, help="a sentence on the subject, helps transcribe jargon")
    p.set_defaults(func=_cmd_prepare)

    p = sub.add_parser("save", help="store a voice from a reference clip and its exact transcript")
    p.add_argument("name")
    p.add_argument("--clip", type=Path, required=True, help="reference clip (ideally 8-15 s, one voice)")
    p.add_argument("--text", required=True, help="exactly what is said in the clip")
    p.add_argument("--language", default="es", help="language code (default: es)")
    p.add_argument("--profile", type=Path, default=None,
                   help="speaker profile_X.npy from `prepare`, when the clip is not one of its candidates")
    p.add_argument("--model", choices=["1.7b", "0.6b"], default="1.7b",
                   help="Qwen3-TTS size: 1.7b sounds closer, 0.6b is ~2x faster (default: 1.7b)")
    p.set_defaults(func=_cmd_save)

    p = sub.add_parser("speak", help="speak a text or a [A]/[B] dialogue with stored voices")
    p.add_argument("--voice", default=None, help="voice for text without a [Name] tag")
    p.add_argument("--text", default=None)
    p.add_argument("--file", type=Path, default=None, help="UTF-8 text file (may contain [Name] tags)")
    p.add_argument("--out", type=Path, required=True, help="output WAV")
    p.add_argument("--takes", type=int, default=1,
                   help="generate each sentence group N times and keep the most faithful take (default: 1)")
    p.add_argument("--check", action=argparse.BooleanOptionalAction, default=None,
                   help="reject takes that skip or invent words (default: on when --takes > 1)")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_speak)

    p = sub.add_parser("transcribe", help="timed script of a recording, one block per stretch of speech")
    p.add_argument("recording", type=Path)
    p.add_argument("--speakers", type=int, default=2, help="how many different voices (default: 2)")
    p.add_argument("--language", default="es", help="language spoken in the recording (default: es)")
    p.add_argument("--topic", default=None, help="a sentence on the subject, helps transcribe jargon")
    p.set_defaults(func=_cmd_transcribe)

    p = sub.add_parser("dub", help="speak a translated script with the speakers' voices, at the original times")
    p.add_argument("recording", type=Path)
    p.add_argument("--script", type=Path, required=True, help="translated script (same [n] headers)")
    p.add_argument("--language", default="en", help="language of the script (default: en)")
    p.add_argument("--out", type=Path, default=None, help="default: output/<name>/dub/<name>_<language>.wav")
    p.add_argument("--takes", type=int, default=3, help="at most N takes per sentence group (default: 3)")
    p.add_argument("--good-enough", type=float, default=None,
                   help="stop trying more takes once one is complete and at least this alike")
    p.add_argument("--seed", type=int, default=0,
                   help="another seed gives different takes; to redo a block, delete its files in blocks_<language>/")
    p.set_defaults(func=_cmd_dub)

    p = sub.add_parser("list", help="list the stored voices")
    p.set_defaults(func=_cmd_list)
    return parser


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except UserError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0
