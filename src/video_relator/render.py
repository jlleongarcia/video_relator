"""Combine narration + music into the soundtrack and put it on the video."""

from video_relator import ffmpeg
from video_relator.errors import UserError
from video_relator.mix import RATE, build
from video_relator.wavio import read_wav, write_wav
from video_relator.workspace import Workspace


def make_soundtrack(ws: Workspace) -> None:
    for path, command in ((ws.voice, "narrate"), (ws.music, "music")):
        if not path.exists():
            raise UserError(f"{path} not found. Run: video-relator {command} {ws.video}")
    voice, voice_rate = read_wav(ws.voice)
    music, music_rate = read_wav(ws.music)
    if voice_rate != RATE or music_rate != RATE:
        raise UserError(f"Expected {RATE} Hz audio, got {voice_rate} (voice) and {music_rate} (music)")

    raw = ws.dir / "soundtrack_raw.wav"
    write_wav(raw, build(voice, music), RATE)
    ffmpeg.loudnorm(raw, ws.soundtrack)
    raw.unlink()


def make_videos(ws: Workspace, *, h264: bool, original: bool, preview_seconds: float | None = None) -> list:
    """Write the requested versions of the video with the new soundtrack; returns their paths."""
    written = []
    for wanted, is_h264 in ((original, False), (h264, True)):
        if wanted:
            dest = ws.narrated_video(h264=is_h264, preview=preview_seconds is not None)
            ffmpeg.mux(ws.video, ws.soundtrack, dest, h264=is_h264, seconds=preview_seconds)
            written.append(dest)
    return written
