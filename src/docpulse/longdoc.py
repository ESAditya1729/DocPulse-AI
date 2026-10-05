"""Chunked map-reduce preparation of large documents.

Replaces a silent raw_text[:16000] truncation with: documents that already
fit are passed through unchanged (same cost as today), and documents that
don't are split into overlapping chunks, each condensed by an injected
`generate` callback, then stitched back into one block of notes covering the
whole document instead of just its first few thousand characters.
"""

from collections.abc import Callable
from dataclasses import dataclass

DIRECT_THRESHOLD = 16000  # unchanged from the previous hard truncation limit
CHUNK_SIZE = 12000
CHUNK_OVERLAP = 300
MAX_CHUNKS = 20  # hard cap so a pathological file can't trigger unbounded LLM calls


@dataclass
class PreparedContent:
    text: str
    chunked: bool
    chunk_count: int
    truncated: bool  # hit MAX_CHUNKS; trailing content was dropped


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into chunks on paragraph breaks, each up to chunk_size.

    Each chunk after the first carries the last `overlap` characters of the
    previous chunk prepended, for continuity across the boundary. A single
    paragraph longer than chunk_size is hard-split by character count.
    """
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    buffer = ""

    def flush() -> None:
        nonlocal buffer
        if buffer:
            chunks.append(buffer)
            buffer = ""

    for para in paragraphs:
        candidate = f"{buffer}\n\n{para}" if buffer else para

        if len(candidate) <= chunk_size:
            buffer = candidate
            continue

        flush()
        if len(para) <= chunk_size:
            buffer = para
        else:
            for start in range(0, len(para), chunk_size):
                chunks.append(para[start : start + chunk_size])

    flush()

    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    with_overlap = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-overlap:]
        with_overlap.append(tail + chunks[i])
    return with_overlap


def prepare_content(
    generate: Callable[[str, int, int], str],
    raw_text: str,
    *,
    on_progress: Callable[[int, int], None] | None = None,
) -> PreparedContent:
    """Return a safe, full-coverage text representation of raw_text.

    For documents within DIRECT_THRESHOLD, returns raw_text unchanged and
    never calls `generate`. For larger documents, splits into chunks (capped
    at MAX_CHUNKS), condenses each via `generate(chunk, index, total)`, and
    joins the notes into one block.
    """
    if len(raw_text) <= DIRECT_THRESHOLD:
        return PreparedContent(text=raw_text, chunked=False, chunk_count=1, truncated=False)

    chunks = chunk_text(raw_text)
    truncated = len(chunks) > MAX_CHUNKS
    chunks = chunks[:MAX_CHUNKS]
    total = len(chunks)

    notes = []
    for index, chunk in enumerate(chunks, start=1):
        if on_progress:
            on_progress(index, total)
        notes.append(f"[Excerpt {index}/{total}]\n{generate(chunk, index, total)}")

    return PreparedContent(
        text="\n\n".join(notes),
        chunked=True,
        chunk_count=total,
        truncated=truncated,
    )
