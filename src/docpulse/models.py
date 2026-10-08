"""Core domain and analysis models for DocPulse."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Document representation models
# ---------------------------------------------------------------------------


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
    """Represents a fully parsed, normalized document."""

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


# ---------------------------------------------------------------------------
# Analysis Models (Pydantic for clean schema validation & JSON serialization)
# ---------------------------------------------------------------------------


class ConceptItem(BaseModel):
    """A key concept extracted from a document."""

    name: str
    description: str = ""
    importance: str = Field(default="medium", description="high, medium, or low")
    sections: list[int | str] = Field(default_factory=list)
    related_concepts: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)


class ConceptIndexResult(BaseModel):
    """Structured collection of document concepts."""

    document: str
    concepts: list[ConceptItem] = Field(default_factory=list)


class PrerequisiteItem(BaseModel):
    """A prerequisite knowledge requirement."""

    name: str
    importance: str = Field(default="medium", description="high, medium, or low")
    difficulty: str = Field(default="medium", description="beginner, medium, advanced, or high")
    needed_for: str = ""
    relevant_sections: list[int | str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)


class PrerequisiteResult(BaseModel):
    """Structured collection of prerequisites and their relationships."""

    document: str
    prerequisites: list[PrerequisiteItem] = Field(default_factory=list)


class EquationVariable(BaseModel):
    """A variable definition in an equation."""

    symbol: str
    description: str


class EquationItem(BaseModel):
    """A mathematical equation identified in a document."""

    id: int | str
    latex: str = ""
    readable: str = ""
    section: str | int = ""
    variables: list[EquationVariable] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    explanation: str = ""
    usage: str = ""


class EquationResult(BaseModel):
    """Structured collection of equations from a document."""

    document: str
    equations: list[EquationItem] = Field(default_factory=list)


class DocumentMapSection(BaseModel):
    """Summary of a section in the document map."""

    id: int
    title: str
    level: int = 1
    page: int | None = None


class DocumentMapResult(BaseModel):
    """Structured unified map of a document."""

    document: str
    sections: list[DocumentMapSection] = Field(default_factory=list)
    core_concepts: list[ConceptItem] = Field(default_factory=list)
    prerequisites: list[PrerequisiteItem] = Field(default_factory=list)
    equations: list[EquationItem] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)


class FlashcardItem(BaseModel):
    """A flashcard question and answer pair."""

    question: str
    answer: str


class StudyQuestionItem(BaseModel):
    """A study question for self-assessment."""

    question: str
    type: str = Field(default="conceptual", description="conceptual, multiple_choice, or short_answer")
    options: list[str] = Field(default_factory=list)
    answer: str = ""
    explanation: str = ""


class StudyModeResult(BaseModel):
    """Structured learning guide and study material."""

    document: str
    key_concepts: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    important_equations: list[str] = Field(default_factory=list)
    flashcards: list[FlashcardItem] = Field(default_factory=list)
    questions: list[StudyQuestionItem] = Field(default_factory=list)


class DocumentComparisonResult(BaseModel):
    """Structured comparison between two documents."""

    document_a: str
    document_b: str
    shared_concepts: list[str] = Field(default_factory=list)
    unique_to_a: list[str] = Field(default_factory=list)
    unique_to_b: list[str] = Field(default_factory=list)
    methodology_a: str = ""
    methodology_b: str = ""
    prerequisites_comparison: str = ""
    equations_comparison: str = ""
    key_differences: list[str] = Field(default_factory=list)
    conclusion: str = ""


class CitationEvidence(BaseModel):
    """Citation location evidence for an answer."""

    section: str | int | None = None
    section_title: str | None = None
    page: int | None = None
    excerpt: str | None = None


class AnswerWithEvidence(BaseModel):
    """Answer with optional citation grounding."""

    answer: str
    evidence: list[CitationEvidence] = Field(default_factory=list)
    was_cached: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
