"""Thin wrappers around the ffmpeg binary."""

import json
import re
import shutil
import subprocess
from pathlib import Path

from video_relator.errors import UserError


def ffmpeg_exe() -> str:
    """Use the ffmpeg on PATH if there is one, else the binary bundled with imageio-ffmpeg."""
    system = shutil.which("ffmpeg")
    if system:
        return system
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def run(*args: str) -> None:
    """Run ffmpeg with the given arguments, raising if it fails."""
    subprocess.run([ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", *args], check=True)


def extract_audio(source: Path, dest: Path, *, sample_rate: int = 16000, mono: bool = True) -> None:
    """Extract the audio track of a video (or audio) file to a WAV file.

    Defaults (16 kHz mono) are what speech recognition expects.
    """
    channels = ["-ac", "1"] if mono else []
    run("-i", str(source), "-vn", *channels, "-ar", str(sample_rate), str(dest))


def duration(path: Path) -> float:
    """Length in seconds of a media file."""
    info = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    found = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", info.stderr)
    if not found:
        raise UserError(f"Could not read the duration of {path}")
    hours, minutes, seconds = found.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def scene_changes(video: Path, *, threshold: float = 0.15) -> list[float]:
    """Times (seconds) where the picture changes noticeably. Decodes the whole video, so it is slow."""
    log = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(video), "-an",
         "-vf", f"scale=320:-1,select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
        capture_output=True, text=True).stderr
    return sorted({round(float(t), 2) for t in re.findall(r"pts_time:([0-9.]+)", log)})


def import_music(source: Path, dest: Path, seconds: float, *, fade_in: float = 4.0, fade_out: float = 8.0) -> None:
    """Convert any audio file to 48 kHz stereo WAV of the given length (looped or cut) with fades."""
    fades = f"afade=t=in:d={fade_in},afade=t=out:st={max(0.0, seconds - fade_out)}:d={fade_out}"
    run("-stream_loop", "-1", "-i", str(source), "-t", f"{seconds:.3f}", "-vn",
        "-af", fades, "-ar", "48000", "-ac", "2", str(dest))


def loudnorm(source: Path, dest: Path) -> None:
    """Bring the overall loudness to -16 LUFS (the usual target for online video).

    Two passes: measure first, then apply one constant gain. The one-pass mode adapts as it goes
    and boosts a quiet intro before it has heard the louder speech.
    """
    target = "I=-16:TP=-1.5:LRA=11"
    measure = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(source), "-af", f"loudnorm={target}:print_format=json",
         "-f", "null", "-"], capture_output=True, text=True, check=True)
    stats = json.loads(re.findall(r"\{[^{}]*\}", measure.stderr)[-1])
    measured = (f"measured_I={stats['input_i']}:measured_TP={stats['input_tp']}:"
                f"measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}:"
                f"offset={stats['target_offset']}")
    run("-i", str(source), "-af", f"loudnorm={target}:{measured}:linear=true", "-ar", "48000", str(dest))


def mux(video: Path, audio: Path, dest: Path, *, h264: bool, seconds: float | None = None) -> None:
    """Put a new audio track on a video.

    h264=False copies the video stream untouched (same codec, same quality, same size);
    h264=True re-encodes it to a much smaller H.264 file that plays practically everywhere.
    seconds limits the output to its first N seconds (for quick previews).
    """
    video_codec = ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p"] if h264 \
        else ["-c:v", "copy"]
    limit = ["-t", f"{seconds:.3f}"] if seconds else []
    run("-i", str(video), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", *video_codec,
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", *limit, "-shortest", str(dest))
