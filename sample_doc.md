# DocPulse Architecture & System Design

## 1. Overview
DocPulse is an intelligent developer CLI tool for automated document understanding, prerequisites discovery, section-level indexing, and study guide synthesis.

## 2. Core Architecture
The system is built on a 3-tier architecture:
- **Presentation Layer**: Typer and Rich CLI interface for terminal formatting and interactive prompts.
- **Parsing & Indexing Layer**: Extensible parser modules for PDF and Markdown files.
- **Cognitive Engine**: IBM Consulting Advantage (ICA) Developer Gateway client for LLM processing.

## 3. Key Modules
- `config.py`: Persistent configuration and credentials management.
- `llm.py`: OpenAPI-compliant HTTP client with resilient error extraction.
- `parsers`: Modular document structure extraction with section ID assignment.
- `cli.py`: User-facing commands (`init`, `analyze`, `section`, `sources`, `export`).

## 4. Dependencies & References
- Typer: https://typer.tiangolo.com
- Rich: https://github.com/Textualize/rich
- HTTPX: https://www.python-httpx.org
- PyPDF: https://pypdf.readthedocs.io
