"""Document structure models and parser definitions."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class Section:
    """Represents an indexed section of a document."""
    id: int
    title: str
    content: str
    level: int = 1
    page_number: int | None = None


@dataclass
class ParsedDocument:
    """Represents a fully parsed document."""
    file_path: Path
    file_name: str
    doc_type: str
    sections: list[Section] = field(default_factory=list)
    raw_text: str = ""

    def get_section_by_id(self, sec_id: int) -> Section | None:
        for sec in self.sections:
            if sec.id == sec_id:
                return sec
        return None

    def get_table_of_contents(self) -> list[tuple[int, str, int]]:
        """Returns list of (id, title, level)."""
        return [(s.id, s.title, s.level) for s in self.sections]


class DocumentParser(Protocol):
    """Protocol for modular document parsers."""
    def parse(self, file_path: Path) -> ParsedDocument:
        ...
