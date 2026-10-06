"""Command line entry point: `video-relator <command> ...`."""

import argparse
import sys
from pathlib import Path

from video_relator.errors import UserError
from video_relator.workspace import DEFAULT_OUTPUT_ROOT, Workspace


def _workspace(args: argparse.Namespace) -> Workspace:
    if not args.video.exists():
        raise UserError(f"{args.video} not found")
    return Workspace(args.video, args.output)


def _cmd_transcribe(args: argparse.Namespace) -> None:
    from video_relator.transcribe import transcribe

    ws = _workspace(args)
    if ws.transcript_json.exists() and not args.force:
        print(f"Already transcribed: {ws.dir} (use --force to redo)")
        return

    result = transcribe(ws, model_size=args.model, language=args.language)
    print(f"Language: {result['language']} (p={result['language_probability']})")
    print(f"{len(result['segments'])} segments written to {ws.dir}")


def _cmd_narrate(args: argparse.Namespace) -> None:
    from video_relator import narrate

    ws = _workspace(args)
    script = args.script or ws.script
    if not script.exists():
        raise UserError(f"Narration script {script} not found (see README for its format)")
    if args.sample:
        narrate.sample(ws, script, args.sample, args.voice)
    else:
        narrate.narrate(ws, script, args.voice)
        print(f"Narration written to {ws.voice}")


def _cmd_music(args: argparse.Namespace) -> None:
    from video_relator import music

    ws = _workspace(args)
    path = music.make(ws, seed=args.seed, mood=args.mood, file=args.file)
    print(f"Music written to {path}" + ("" if args.file else f" (mood: {args.mood})"))


def _cmd_render(args: argparse.Namespace) -> None:
    from video_relator import render

    ws = _workspace(args)
    render.make_soundtrack(ws)
    print(f"Soundtrack written to {ws.soundtrack}")
    if args.audio_only:
        return
    for path in render.make_videos(ws, original=args.formats in ("original", "both"),
                                   h264=args.formats in ("h264", "both"), preview_seconds=args.preview):
        print(f"Video written to {path}")


def _add_video_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("video", type=Path, help="path to the video file")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT,
                   help=f"root folder for results (default: {DEFAULT_OUTPUT_ROOT})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-relator",
        description="Transcribe a video's narration and re-narrate it with synthetic voice and music.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("transcribe", help="transcribe the narration of a video")
    _add_video_arguments(p)
    p.add_argument("--model", default="small",
                   help="Whisper model: tiny, base, small, medium, large-v3 (default: small)")
    p.add_argument("--language", default=None,
                   help="language code such as es or en (default: auto-detect)")
    p.add_argument("--force", action="store_true", help="redo even if a transcript exists")
    p.set_defaults(func=_cmd_transcribe)

    p = sub.add_parser("narrate", help="speak the narration script with a synthetic voice")
    _add_video_arguments(p)
    p.add_argument("--script", type=Path, default=None,
                   help="narration script (default: <output>/<video-name>/narration.txt)")
    p.add_argument("--voice", default=None,
                   help="Kokoro voice, e.g. ef_dora, em_alex (default: chosen from the video's language)")
    p.add_argument("--sample", type=int, default=None, metavar="N",
                   help="only speak block N, to audition the voice")
    p.set_defaults(func=_cmd_narrate)

    p = sub.add_parser("music", help="create the background music")
    _add_video_arguments(p)
    from video_relator.music import DEFAULT_MOOD, MOODS

    p.add_argument("--mood", choices=list(MOODS), default=DEFAULT_MOOD,
                   help="style to match the topic. " + "; ".join(f"{k}: {m.description}" for k, m in MOODS.items()))
    p.add_argument("--seed", type=int, default=7, help="different seed = different music (default: 7)")
    p.add_argument("--file", type=Path, default=None,
                   help="use this audio file (looped/cut to the video length) instead of generated music")
    p.set_defaults(func=_cmd_music)

    p = sub.add_parser("render", help="mix voice and music and put them on the video")
    _add_video_arguments(p)
    p.add_argument("--audio-only", action="store_true", help="only build the soundtrack, not the videos")
    p.add_argument("--formats", choices=["original", "h264", "both"], default="both",
                   help="original = same video stream, new audio; h264 = smaller re-encoded file (default: both)")
    p.add_argument("--preview", type=float, default=None, metavar="SECONDS",
                   help="only the first SECONDS of the video, for a quick check")
    p.set_defaults(func=_cmd_render)
    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a legacy codepage; transcripts can be in any language.
    sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except UserError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0
