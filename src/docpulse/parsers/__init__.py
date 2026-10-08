"""Parser loader and factory for DocPulse."""

from pathlib import Path

from docpulse.errors import DocumentError
from docpulse.parsers.base import DocumentParser, ParsedDocument, Section
from docpulse.parsers.markdown_parser import MarkdownParser
from docpulse.parsers.pdf_parser import PDFParser

MARKDOWN_SUFFIXES = frozenset({".md", ".markdown", ".txt"})
PDF_SUFFIXES = frozenset({".pdf"})
SUPPORTED_SUFFIXES = MARKDOWN_SUFFIXES | PDF_SUFFIXES

SUPPORTED_TYPES_HINT = (
    "Supported document types: " + ", ".join(sorted(SUPPORTED_SUFFIXES)) + ".\n"
    "Pass --force-text to read any file as plain text, or convert it to Markdown/PDF first."
)


def get_parser(file_path: Path | str, *, force_text: bool = False) -> DocumentParser:
    """Select the appropriate parser based on file extension.

    Unknown extensions are rejected rather than silently falling back to the
    text parser: a .docx or .epub read as UTF-8 yields mojibake that the
    analyzers then confidently report as findings.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if force_text:
        return MarkdownParser()
    if suffix in MARKDOWN_SUFFIXES:
        return MarkdownParser()
    if suffix in PDF_SUFFIXES:
        return PDFParser()
    if not suffix:
        # Extension-less files are conventionally plain text.
        return MarkdownParser()

    raise DocumentError(f"Unsupported document type: '{suffix}'", hint=SUPPORTED_TYPES_HINT)


def parse_document(file_path: Path | str, *, force_text: bool = False) -> ParsedDocument:
    """Parse a document directly given its path, raising DocumentError on failure."""
    path = Path(file_path)
    if not path.exists():
        raise DocumentError(f"Document not found at: {path}")
    if path.is_dir():
        raise DocumentError(f"{path} is a directory, not a document.")

    parser = get_parser(path, force_text=force_text)
    try:
        parsed = parser.parse(path)
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError(f"Could not parse {path.name} ({type(exc).__name__}: {exc})") from exc

    if not parsed.raw_text.strip():
        hint = "Scanned PDFs need OCR before they can be analyzed."
        if path.suffix.lower() != ".pdf":
            hint += " To read an unsupported file as plain text, put --force-text before the command: 'docpulse --force-text analyze <file>'."
        raise DocumentError(
            f"{path.name} contains no extractable text.",
            hint=hint,
        )
    return parsed


__all__ = [
    "DocumentParser",
    "MarkdownParser",
    "PDFParser",
    "ParsedDocument",
    "Section",
    "get_parser",
    "parse_document",
    "SUPPORTED_SUFFIXES",
]