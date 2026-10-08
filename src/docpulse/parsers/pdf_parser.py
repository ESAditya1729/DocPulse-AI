"""PDF document parser with page/heading-aware section indexing."""

import re
from pathlib import Path

from pypdf import PdfReader

from docpulse.parsers.base import DocumentParser, ParsedDocument, Section


class PDFParser(DocumentParser):
    """Parses PDF documents into structured sections."""

    def parse(self, file_path: Path) -> ParsedDocument:
        reader = PdfReader(str(file_path))
        full_text_list = []
        sections: list[Section] = []
        sec_id = 1

        for page_idx, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            full_text_list.append(page_text)

            # Simple heuristic for section breaks or page-based chunking
            lines = page_text.splitlines()
            current_title = f"Page {page_idx}"
            current_lines = []

            for line in lines:
                stripped = line.strip()
                # Check for major heading pattern: e.g. "1. Introduction" or all caps short title
                if (
                    re.match(r"^(?:[0-9]+\.|\b[I|V|X]+\.)\s+[A-Z]", stripped)
                    or (stripped.isupper() and 3 < len(stripped) < 50)
                ):
                    if current_lines:
                        sections.append(
                            Section(
                                id=sec_id,
                                title=current_title,
                                content="\n".join(current_lines).strip(),
                                level=1,
                                page_number=page_idx,
                            )
                        )
                        sec_id += 1
                        current_lines = []
                    current_title = stripped
                else:
                    current_lines.append(line)

            if current_lines:
                sections.append(
                    Section(
                        id=sec_id,
                        title=current_title,
                        content="\n".join(current_lines).strip(),
                        level=1,
                        page_number=page_idx,
                    )
                )
                sec_id += 1

        raw_text = "\n\n".join(full_text_list)
        return ParsedDocument(
            file_path=file_path,
            file_name=file_path.name,
            doc_type="pdf",
            sections=sections,
            raw_text=raw_text,
        )
