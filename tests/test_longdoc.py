"""Tests for chunked map-reduce preparation of large documents."""

from docpulse.longdoc import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DIRECT_THRESHOLD,
    MAX_CHUNKS,
    chunk_text,
    prepare_content,
)


def test_chunk_text_single_chunk_when_short():
    text = "para one\n\npara two"
    assert chunk_text(text) == [text]


def test_chunk_text_splits_long_text_with_overlap():
    # Build paragraphs that force at least two chunks.
    paragraphs = [f"Paragraph {i} " + ("x" * 500) for i in range(40)]
    text = "\n\n".join(paragraphs)

    chunks = chunk_text(text, chunk_size=4000, overlap=200)

    assert len(chunks) > 1
    for chunk in chunks:
        # Overlap means a chunk can exceed chunk_size by up to `overlap`.
        assert len(chunk) <= 4000 + 200
    # Second chunk should start with the tail of the first (the overlap).
    assert chunks[1].startswith(chunks[0][-200:])


def test_chunk_text_hard_splits_a_single_oversized_paragraph():
    text = "x" * 10000  # one giant paragraph, no blank lines at all
    chunks = chunk_text(text, chunk_size=3000, overlap=0)
    assert len(chunks) == 4  # 3000 * 3 + 1000
    assert "".join(chunks) == text


def test_prepare_content_skips_generate_for_short_text():
    calls = []

    def generate(chunk, index, total):
        calls.append((chunk, index, total))
        return "should not be called"

    short_text = "a" * (DIRECT_THRESHOLD - 1)
    prepared = prepare_content(generate, short_text)

    assert prepared.text == short_text
    assert prepared.chunked is False
    assert prepared.truncated is False
    assert calls == []


def test_prepare_content_chunks_and_calls_generate_per_chunk():
    calls = []

    def generate(chunk, index, total):
        calls.append((index, total))
        return f"notes-{index}"

    # Force several chunks: paragraphs sized so each is its own chunk under CHUNK_SIZE.
    paragraphs = [("p" * (CHUNK_SIZE - 100)) for _ in range(3)]
    long_text = "\n\n".join(paragraphs)
    assert len(long_text) > DIRECT_THRESHOLD

    progress = []
    prepared = prepare_content(generate, long_text, on_progress=lambda i, n: progress.append((i, n)))

    assert prepared.chunked is True
    assert prepared.truncated is False
    assert prepared.chunk_count == len(calls)
    assert calls == [(i, prepared.chunk_count) for i in range(1, prepared.chunk_count + 1)]
    assert progress == calls
    for i in range(1, prepared.chunk_count + 1):
        assert f"[Excerpt {i}/{prepared.chunk_count}]\nnotes-{i}" in prepared.text


def test_prepare_content_caps_at_max_chunks_and_flags_truncation():
    def generate(chunk, index, total):
        return f"notes-{index}"

    # Enough distinct paragraphs to produce far more than MAX_CHUNKS chunks.
    paragraphs = [("p" * (CHUNK_SIZE - 100)) for _ in range(MAX_CHUNKS + 10)]
    huge_text = "\n\n".join(paragraphs)

    call_count = 0
    original_generate = generate

    def counting_generate(chunk, index, total):
        nonlocal call_count
        call_count += 1
        return original_generate(chunk, index, total)

    prepared = prepare_content(counting_generate, huge_text)

    assert prepared.truncated is True
    assert prepared.chunk_count == MAX_CHUNKS
    assert call_count == MAX_CHUNKS


def test_overlap_constant_is_smaller_than_chunk_size():
    # Sanity check on the module defaults themselves.
    assert 0 < CHUNK_OVERLAP < CHUNK_SIZE
