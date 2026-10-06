"""Per-video working folder: every artefact for one video lives in output/<video-name>/."""

from dataclasses import dataclass
from pathlib import Path

DEFAULT_OUTPUT_ROOT = Path("output")


@dataclass(frozen=True)
class Workspace:
    video: Path
    root: Path = DEFAULT_OUTPUT_ROOT

    @property
    def dir(self) -> Path:
        return self.root / self.video.stem

    @property
    def audio(self) -> Path:
        """Original audio track, extracted for analysis."""
        return self.dir / "audio.wav"

    @property
    def transcript_json(self) -> Path:
        """Full transcript with segment timings (machine-readable)."""
        return self.dir / "transcript.json"

    @property
    def transcript_txt(self) -> Path:
        """Plain transcript, one segment per line."""
        return self.dir / "transcript.txt"

    @property
    def transcript_timed(self) -> Path:
        """Transcript with a time range in front of each segment."""
        return self.dir / "transcript_timed.txt"

    @property
    def cuts(self) -> Path:
        """Times where the picture changes (cached: finding them takes a minute)."""
        return self.dir / "cuts.json"

    @property
    def script(self) -> Path:
        """Default narration script (see script.py); --script can point elsewhere."""
        return self.dir / "narration.txt"

    @property
    def voice(self) -> Path:
        """New narration, 48 kHz mono, as long as the video, silent where nobody speaks."""
        return self.dir / "voice.wav"

    @property
    def timeline(self) -> Path:
        """Where each block of the new narration starts and ends."""
        return self.dir / "voice_timeline.json"

    @property
    def samples_dir(self) -> Path:
        return self.dir / "samples"

    @property
    def music(self) -> Path:
        """Background music, 48 kHz stereo, as long as the video."""
        return self.dir / "music.wav"

    @property
    def soundtrack(self) -> Path:
        """Final mix of voice and music."""
        return self.dir / "soundtrack.wav"

    def narrated_video(self, *, h264: bool, preview: bool = False) -> Path:
        suffix = ("_h264" if h264 else "") + ("_preview" if preview else "")
        return self.dir / f"{self.video.stem}_narrated{suffix}.mp4"

    def ensure(self) -> "Workspace":
        self.dir.mkdir(parents=True, exist_ok=True)
        return self
