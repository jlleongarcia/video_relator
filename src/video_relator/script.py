"""The narration script: the text file the synthetic voice reads.

Format: numbered blocks, one per stretch of speech in the original video.

    [1]
    First paragraph to be spoken.

    [2]
    Second paragraph.

A `[?]` anywhere in a block marks a doubtful word; it is never spoken, only reported.
A `[pause 1.5]` marker adds a silence (in seconds) at that exact spot, for emphasis.
Punctuation also shapes the rhythm: a full stop pauses longer than a comma.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from video_relator.errors import UserError

_HEADER = re.compile(r"^\[(\d+)\]$")
_DOUBT = re.compile(r"\s*\[\?\]")


@dataclass(frozen=True)
class Block:
    number: int
    text: str
    doubtful: bool


def load_script(path: Path) -> list[Block]:
    blocks: list[Block] = []
    number: int | None = None
    lines: list[str] = []

    def close() -> None:
        if number is None:
            return
        raw = " ".join(lines)
        text = _DOUBT.sub("", raw).strip()
        if not text:
            raise UserError(f"{path}: block [{number}] is empty")
        blocks.append(Block(number, text, doubtful="[?]" in raw))

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        header = _HEADER.match(line)
        if header:
            close()
            number, lines = int(header.group(1)), []
        elif line:
            if number is None:
                raise UserError(f"{path}: text found before the first [n] header")
            lines.append(line)
    close()

    if not blocks:
        raise UserError(f"{path}: no [n] blocks found")
    return blocks
