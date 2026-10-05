# DocPulse

`docpulse` is a fast, modern CLI tool for intelligent document understanding, prerequisites analysis, section indexing, reference tracking, and study guide generation powered by LLMs (IBM ICA Gateway).

## Key Features

- **`docpulse init`**: Interactive connection setup and health-check verification.
- **`docpulse analyze <doc>`**: Extract prerequisites, core themes, key concepts, and summaries.
- **`docpulse section <doc> --id <sec>`**: Deep-dive focused summaries of specific sections.
- **`docpulse sources <doc>`**: Identify cited papers, external links, and referenced repositories.
- **`docpulse export <doc> --format md`**: Generate structured study guides or markdown notes.
- **`docpulse ask <doc> ["question"]`**: Ask follow-up questions about a document — one-shot if you
  pass a question, or an interactive back-and-forth session if you don't.

Equations in LLM output are rendered as their own boxed panel (standalone `$$...$$` equations) or
as clean Unicode notation inline (`$...$` math inside a sentence), instead of raw LaTeX.

## Installation

```bash
pip install -e .
```

## Quickstart

```bash
# 1. Initialize and configure ICA Gateway credentials
docpulse init

# 2. Analyze a document
docpulse analyze paper.pdf

# 3. Inspect a specific section
docpulse section paper.pdf --id 2

# 4. Extract external resources
docpulse sources guide.md

# 5. Export a study guide
docpulse export paper.pdf --format md --output study_guide.md

# 6. Ask a one-off question, or omit the question for an interactive session
docpulse ask paper.pdf "What assumptions does this rely on?"
docpulse ask paper.pdf
```

## Large documents

`analyze`, `section`, `sources`, `export`, and `ask` all cover the **whole** document, not just the
first ~16,000 characters. Documents under that size are sent as-is; larger ones are split into
overlapping chunks, each condensed by the LLM, then combined into one block before the final
request — so you'll see a `Summarizing excerpt i/n...` notice rather than a silent cutoff. There's a
hard cap of 20 chunks (~240,000 characters); if a document exceeds that, you'll get an explicit
warning instead of a silent truncation.

## Response caching

Every LLM call is cached locally, keyed on the exact request (endpoint, namespace, model, prompt,
and messages) — re-running the same command on an unchanged document is instant and free, and
you'll see a `Served from cache` notice. Pass `--no-cache` to any command to force a fresh call.
The cache lives at `~/.docpulse/cache` (override with `DOCPULSE_CACHE_DIR`).
