"""Parser loader and factory for DocPulse."""

from pathlib import Path

from docpulse.parsers.base import DocumentParser, ParsedDocument, Section
from docpulse.parsers.markdown_parser import MarkdownParser
from docpulse.parsers.pdf_parser import PDFParser


def get_parser(file_path: Path | str) -> DocumentParser:
    """Select the appropriate parser based on file extension."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in (".md", ".markdown", ".txt"):
        return MarkdownParser()
    elif suffix == ".pdf":
        return PDFParser()
    else:
        # Fallback to text/markdown parser
        return MarkdownParser()


def parse_document(file_path: Path | str) -> ParsedDocument:
    """Parse a document directly given its path."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found at: {path}")
    parser = get_parser(path)
    return parser.parse(path)


__all__ = ["DocumentParser", "MarkdownParser", "PDFParser", "ParsedDocument", "Section", "get_parser", "parse_document"]
