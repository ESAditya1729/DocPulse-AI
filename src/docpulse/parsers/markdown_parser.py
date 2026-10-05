"""Markdown document parser with heading-based section indexing."""

import re
from pathlib import Path

from docpulse.parsers.base import DocumentParser, ParsedDocument, Section


class MarkdownParser(DocumentParser):
    """Parses Markdown files into structured sections based on headers."""

    def parse(self, file_path: Path) -> ParsedDocument:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            content = f.read()

        lines = content.splitlines()
        sections: list[Section] = []
        current_title = "Introduction"
        current_level = 1
        current_lines: list[str] = []
        sec_id = 1

        heading_pattern = re.compile(r"^(#{1,6})\s+(.*)$")

        for line in lines:
            match = heading_pattern.match(line)
            if match:
                # Flush previous section if it has content
                if current_lines or sec_id > 1:
                    sec_text = "\n".join(current_lines).strip()
                    if sec_text or sec_id > 1:
                        sections.append(
                            Section(
                                id=sec_id,
                                title=current_title,
                                content=sec_text,
                                level=current_level,
                            )
                        )
                        sec_id += 1
                        current_lines = []

                hashes, title = match.groups()
                current_level = len(hashes)
                current_title = title.strip()
            else:
                current_lines.append(line)

        # Flush final section
        sec_text = "\n".join(current_lines).strip()
        sections.append(
            Section(
                id=sec_id,
                title=current_title,
                content=sec_text,
                level=current_level,
            )
        )

        return ParsedDocument(
            file_path=file_path,
            file_name=file_path.name,
            doc_type="markdown",
            sections=sections,
            raw_text=content,
        )
