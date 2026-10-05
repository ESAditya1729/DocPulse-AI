# DocPulse

`docpulse` is a fast, modern CLI tool for intelligent document understanding, prerequisites analysis, section indexing, reference tracking, and study guide generation powered by LLMs (IBM ICA Gateway).

## Key Features

- **`docpulse init`**: Interactive connection setup and health-check verification.
- **`docpulse analyze <doc>`**: Extract prerequisites, core themes, key concepts, and summaries.
- **`docpulse section <doc> --id <sec>`**: Deep-dive focused summaries of specific sections.
- **`docpulse sources <doc>`**: Identify cited papers, external links, and referenced repositories.
- **`docpulse export <doc> --format md`**: Generate structured study guides or markdown notes.

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
```
