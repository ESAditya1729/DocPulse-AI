"""Focused analyzers for extracting structured intelligence from documents."""

import json
import re
from typing import Any

from docpulse.mathbox import prettify_math
from docpulse.models import (
    AnswerWithEvidence,
    CitationEvidence,
    ConceptIndexResult,
    ConceptItem,
    DocumentComparisonResult,
    DocumentMapResult,
    DocumentMapSection,
    EquationItem,
    EquationResult,
    EquationVariable,
    FlashcardItem,
    ParsedDocument,
    PrerequisiteItem,
    PrerequisiteResult,
    StudyModeResult,
    StudyQuestionItem,
)

# Reported when a structured command gets a response it could not read as JSON.
# Without this the command used to print "No concepts identified", which reads
# as a fact about the document rather than a failure of the extraction.
UNPARSEABLE_JSON_WARNING = (
    "Model response was not parseable JSON, so the extracted result may be empty or incomplete."
)


def _collect_warnings(data: Any, raw_response: str, warnings: list[str] | None, message: str) -> None:
    """Append `message` to warnings when the response yielded nothing usable."""
    if warnings is not None and raw_response.strip() and data is None:
        warnings.append(message)


def _extract_json_block(text: str) -> Any:
    """Robustly extract and parse JSON from an LLM response.

    Handles raw JSON, ```json ... ``` code fences, and leading/trailing text.
    """
    text = text.strip()
    # 1. Direct parse attempt
    try:
        return json.loads(text)
    except Exception:  # noqa: S110
        pass

    # 2. Extract from markdown code blocks
    fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    matches = fence_pattern.findall(text)
    for match in matches:
        try:
            return json.loads(match.strip())
        except Exception:  # noqa: S112
            continue

    # 3. Find bracket-delimited spans { ... } or [ ... ]
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace : last_brace + 1])
        except Exception:  # noqa: S110
            pass

    first_bracket = text.find("[")
    last_bracket = text.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        try:
            return json.loads(text[first_bracket : last_bracket + 1])
        except Exception:  # noqa: S110
            pass

    return None


# ---------------------------------------------------------------------------
# Concept Analyzer
# ---------------------------------------------------------------------------


def parse_concepts_response(
    raw_response: str, doc_name: str, warnings: list[str] | None = None
) -> ConceptIndexResult:
    """Parse and validate LLM output for concept index into ConceptIndexResult."""
    data = _extract_json_block(raw_response)
    _collect_warnings(data, raw_response, warnings, UNPARSEABLE_JSON_WARNING)
    concepts: list[ConceptItem] = []

    if isinstance(data, dict):
        raw_list = data.get("concepts", [])
    elif isinstance(data, list):
        raw_list = data
    else:
        raw_list = []

    if raw_list and isinstance(raw_list, list):
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            sections_raw = item.get("sections")
            if isinstance(sections_raw, list):
                sections = sections_raw
            elif sections_raw is not None:
                sections = [sections_raw]
            else:
                sections = []

            related_raw = item.get("related_concepts")
            if isinstance(related_raw, list):
                related = related_raw
            elif related_raw is not None:
                related = [related_raw]
            else:
                related = []

            prereqs_raw = item.get("prerequisites")
            if isinstance(prereqs_raw, list):
                prereqs = prereqs_raw
            elif prereqs_raw is not None:
                prereqs = [prereqs_raw]
            else:
                prereqs = []

            concepts.append(
                ConceptItem(
                    name=name,
                    description=str(item.get("description", "") or "").strip(),
                    importance=str(item.get("importance", "medium") or "medium").strip().lower(),
                    sections=sections,
                    related_concepts=[str(r).strip() for r in related if r is not None and str(r).strip()],
                    prerequisites=[str(p).strip() for p in prereqs if p is not None and str(p).strip()],
                )
            )

    # Fallback heuristic parser if JSON parsing yielded nothing
    if not concepts and raw_response:
        # Heuristic parsing for text lines like "1. Name: Description"
        for line in raw_response.splitlines():
            line = line.strip()
            m = re.match(r"^(?:\d+[\.\)]|\-|\*)\s+\*?\*?([^\*:]+)\*?\*?:\s*(.*)$", line)
            if m:
                cname = m.group(1).strip()
                cdesc = m.group(2).strip()
                if cname:
                    concepts.append(ConceptItem(name=cname, description=cdesc, importance="medium"))

    if not concepts and warnings is not None and raw_response.strip():
        warnings.append("No concepts could be extracted from the model response.")

    return ConceptIndexResult(document=doc_name, concepts=concepts)


# ---------------------------------------------------------------------------
# Prerequisite Analyzer
# ---------------------------------------------------------------------------


def parse_prerequisites_response(
    raw_response: str, doc_name: str, warnings: list[str] | None = None
) -> PrerequisiteResult:
    """Parse and validate LLM output for prerequisites into PrerequisiteResult."""
    data = _extract_json_block(raw_response)
    _collect_warnings(data, raw_response, warnings, UNPARSEABLE_JSON_WARNING)
    prereqs: list[PrerequisiteItem] = []

    if isinstance(data, dict):
        raw_list = data.get("prerequisites", [])
    elif isinstance(data, list):
        raw_list = data
    else:
        raw_list = []

    if raw_list and isinstance(raw_list, list):
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            sections_raw = item.get("relevant_sections") or item.get("sections")
            if isinstance(sections_raw, list):
                sections = sections_raw
            elif sections_raw is not None:
                sections = [sections_raw]
            else:
                sections = []

            deps_raw = item.get("dependencies")
            if isinstance(deps_raw, list):
                deps = deps_raw
            elif deps_raw is not None:
                deps = [deps_raw]
            else:
                deps = []

            prereqs.append(
                PrerequisiteItem(
                    name=name,
                    importance=str(item.get("importance", "medium") or "medium").strip().lower(),
                    difficulty=str(item.get("difficulty", "medium") or "medium").strip().lower(),
                    needed_for=str(item.get("needed_for") or item.get("description") or "").strip(),
                    relevant_sections=sections,
                    dependencies=[str(d).strip() for d in deps if d is not None and str(d).strip()],
                )
            )

    if not prereqs and raw_response:
        for line in raw_response.splitlines():
            line = line.strip()
            m = re.match(r"^(?:\d+[\.\)]|\-|\*)\s+\*?\*?([^\*:]+)\*?\*?:\s*(.*)$", line)
            if m:
                pname = m.group(1).strip()
                pdesc = m.group(2).strip()
                if pname:
                    prereqs.append(PrerequisiteItem(name=pname, needed_for=pdesc))

    if not prereqs and warnings is not None and raw_response.strip():
        warnings.append("No prerequisites could be extracted from the model response.")

    return PrerequisiteResult(document=doc_name, prerequisites=prereqs)


# ---------------------------------------------------------------------------
# Equation Analyzer
# ---------------------------------------------------------------------------


def parse_equations_response(
    raw_response: str, doc_name: str, warnings: list[str] | None = None
) -> EquationResult:
    """Parse and validate LLM output for equations into EquationResult."""
    data = _extract_json_block(raw_response)
    _collect_warnings(data, raw_response, warnings, UNPARSEABLE_JSON_WARNING)
    equations: list[EquationItem] = []

    if isinstance(data, dict):
        raw_list = data.get("equations", [])
    elif isinstance(data, list):
        raw_list = data
    else:
        raw_list = []

    if raw_list and isinstance(raw_list, list):
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            eq_id = item.get("id") or (len(equations) + 1)
            latex = str(item.get("latex") or item.get("equation") or "").strip()
            readable = str(item.get("readable") or "").strip()
            if not readable and latex:
                readable = prettify_math(latex)

            vars_raw = item.get("variables") or []
            vars_list: list[EquationVariable] = []
            if isinstance(vars_raw, list):
                for v in vars_raw:
                    if isinstance(v, dict):
                        sym = str(v.get("symbol") or v.get("name") or "").strip()
                        desc = str(v.get("description") or "").strip()
                        if sym:
                            vars_list.append(EquationVariable(symbol=sym, description=desc))
                    elif isinstance(v, str) and ":" in v:
                        sym, desc = v.split(":", 1)
                        vars_list.append(EquationVariable(symbol=sym.strip(), description=desc.strip()))

            concepts_raw = item.get("concepts")
            if isinstance(concepts_raw, list):
                concepts = concepts_raw
            elif concepts_raw is not None:
                concepts = [concepts_raw]
            else:
                concepts = []

            equations.append(
                EquationItem(
                    id=eq_id,
                    latex=latex,
                    readable=readable,
                    section=item.get("section") or "",
                    variables=vars_list,
                    concepts=[str(c).strip() for c in concepts if c is not None and str(c).strip()],
                    explanation=str(item.get("explanation") or "").strip(),
                    usage=str(item.get("usage") or "").strip(),
                )
            )

    return EquationResult(document=doc_name, equations=equations)


# ---------------------------------------------------------------------------
# Sources / References Analyzer
# ---------------------------------------------------------------------------


def parse_sources_response(raw_response: str, warnings: list[str] | None = None) -> list[str]:
    """Parse and validate LLM output for external references into a flat list."""
    data = _extract_json_block(raw_response)
    _collect_warnings(data, raw_response, warnings, UNPARSEABLE_JSON_WARNING)

    if isinstance(data, dict):
        raw_list = data.get("references", [])
    elif isinstance(data, list):
        raw_list = data
    else:
        raw_list = []

    references: list[str] = []
    if raw_list and isinstance(raw_list, list):
        for item in raw_list:
            if isinstance(item, dict):
                text = str(
                    item.get("citation") or item.get("name") or item.get("url") or item.get("title") or ""
                ).strip()
            elif item is not None:
                text = str(item).strip()
            else:
                text = ""
            if text and text.lower() not in {"none", "none explicitly cited", "n/a"}:
                references.append(text)

    return references


# ---------------------------------------------------------------------------
# Document Map Builder
# ---------------------------------------------------------------------------


def build_document_map_result(
    doc: ParsedDocument,
    concepts: list[ConceptItem],
    prerequisites: list[PrerequisiteItem],
    equations: list[EquationItem],
    references: list[str],
) -> DocumentMapResult:
    """Combine structured analysis pieces into a DocumentMapResult."""
    map_sections = [
        DocumentMapSection(
            id=s.id,
            title=s.title,
            level=s.level,
            page=s.page_number,
        )
        for s in doc.sections
    ]
    return DocumentMapResult(
        document=doc.file_name,
        sections=map_sections,
        core_concepts=concepts,
        prerequisites=prerequisites,
        equations=equations,
        references=references,
    )


# ---------------------------------------------------------------------------
# Document Comparison Analyzer
# ---------------------------------------------------------------------------


def parse_comparison_response(
    raw_response: str, doc_a: str, doc_b: str, warnings: list[str] | None = None
) -> DocumentComparisonResult:
    """Parse and validate LLM output for document comparison into DocumentComparisonResult."""
    data = _extract_json_block(raw_response)
    _collect_warnings(data, raw_response, warnings, UNPARSEABLE_JSON_WARNING)
    if isinstance(data, dict):
        shared_raw = data.get("shared_concepts") or []
        unique_a_raw = data.get("unique_to_a") or []
        unique_b_raw = data.get("unique_to_b") or []
        diffs_raw = data.get("key_differences") or []

        shared = [str(x).strip() for x in (shared_raw if isinstance(shared_raw, list) else [shared_raw]) if x and str(x).strip()]
        unique_a = [str(x).strip() for x in (unique_a_raw if isinstance(unique_a_raw, list) else [unique_a_raw]) if x and str(x).strip()]
        unique_b = [str(x).strip() for x in (unique_b_raw if isinstance(unique_b_raw, list) else [unique_b_raw]) if x and str(x).strip()]
        diffs = [str(x).strip() for x in (diffs_raw if isinstance(diffs_raw, list) else [diffs_raw]) if x and str(x).strip()]

        return DocumentComparisonResult(
            document_a=doc_a,
            document_b=doc_b,
            shared_concepts=shared,
            unique_to_a=unique_a,
            unique_to_b=unique_b,
            methodology_a=str(data.get("methodology_a") or "").strip(),
            methodology_b=str(data.get("methodology_b") or "").strip(),
            prerequisites_comparison=str(data.get("prerequisites_comparison") or "").strip(),
            equations_comparison=str(data.get("equations_comparison") or "").strip(),
            key_differences=diffs,
            conclusion=str(data.get("conclusion") or "").strip(),
        )

    # Fallback for plain markdown or unstructured responses
    return DocumentComparisonResult(
        document_a=doc_a,
        document_b=doc_b,
        conclusion=raw_response.strip(),
    )


# ---------------------------------------------------------------------------
# Study Mode Analyzer
# ---------------------------------------------------------------------------


def parse_study_response(
    raw_response: str, doc_name: str, warnings: list[str] | None = None
) -> StudyModeResult:
    """Parse and validate LLM output for study mode into StudyModeResult."""
    data = _extract_json_block(raw_response)
    _collect_warnings(data, raw_response, warnings, UNPARSEABLE_JSON_WARNING)
    key_concepts: list[str] = []
    prerequisites: list[str] = []
    important_equations: list[str] = []
    flashcards: list[FlashcardItem] = []
    questions: list[StudyQuestionItem] = []

    if isinstance(data, dict):
        kc_raw = data.get("key_concepts") or []
        pr_raw = data.get("prerequisites") or []
        eq_raw = data.get("important_equations") or []
        fc_raw = data.get("flashcards") or []
        q_raw = data.get("questions") or []

        key_concepts = [str(x).strip() for x in (kc_raw if isinstance(kc_raw, list) else [kc_raw]) if x and str(x).strip()]
        prerequisites = [str(x).strip() for x in (pr_raw if isinstance(pr_raw, list) else [pr_raw]) if x and str(x).strip()]
        important_equations = [str(x).strip() for x in (eq_raw if isinstance(eq_raw, list) else [eq_raw]) if x and str(x).strip()]

        if isinstance(fc_raw, list):
            for fc in fc_raw:
                if isinstance(fc, dict) and fc.get("question") and fc.get("answer"):
                    flashcards.append(FlashcardItem(question=str(fc["question"]).strip(), answer=str(fc["answer"]).strip()))

        if isinstance(q_raw, list):
            for q in q_raw:
                if isinstance(q, dict) and q.get("question"):
                    raw_opts = q.get("options")
                    if isinstance(raw_opts, list):
                        opts = [str(o).strip() for o in raw_opts if o is not None and str(o).strip()]
                    elif raw_opts is not None:
                        opts = [str(raw_opts).strip()]
                    else:
                        opts = []

                    questions.append(
                        StudyQuestionItem(
                            question=str(q.get("question")).strip(),
                            type=str(q.get("type") or "conceptual").strip(),
                            options=opts,
                            answer=str(q.get("answer") or "").strip(),
                            explanation=str(q.get("explanation") or "").strip(),
                        )
                    )

    return StudyModeResult(
        document=doc_name,
        key_concepts=key_concepts,
        prerequisites=prerequisites,
        important_equations=important_equations,
        flashcards=flashcards,
        questions=questions,
    )


# ---------------------------------------------------------------------------
# Citation / Evidence Parser
# ---------------------------------------------------------------------------


def parse_answer_with_citations(
    raw_response: str, parsed_doc: ParsedDocument | None = None, warnings: list[str] | None = None
) -> AnswerWithEvidence:
    """Extract answer text and structured evidence citations from response."""
    data = _extract_json_block(raw_response)
    _collect_warnings(data, raw_response, warnings, UNPARSEABLE_JSON_WARNING)
    if isinstance(data, dict) and "answer" in data:
        answer = str(data.get("answer", "") or "").strip()
        ev_raw = data.get("evidence") or []
        ev_list: list[CitationEvidence] = []
        if isinstance(ev_raw, list):
            for ev in ev_raw:
                if isinstance(ev, dict):
                    ev_list.append(
                        CitationEvidence(
                            section=ev.get("section"),
                            section_title=ev.get("section_title"),
                            page=ev.get("page"),
                            excerpt=ev.get("excerpt"),
                        )
                    )
        return AnswerWithEvidence(answer=answer, evidence=ev_list)

    # If raw response is formatted with markdown sections "## Answer" and "## Evidence"
    if "## Evidence" in raw_response or "# Evidence" in raw_response:
        parts = re.split(r"#+\s*Evidence", raw_response, flags=re.IGNORECASE)
        answer = parts[0].strip()
        evidence_text = parts[1].strip() if len(parts) > 1 else ""
        ev_list = []
        for line in evidence_text.splitlines():
            line = line.strip()
            if line.startswith("-") or line.startswith("*") or line.startswith("•"):
                ev_list.append(CitationEvidence(excerpt=line.lstrip("-*• ")))
        return AnswerWithEvidence(answer=answer, evidence=ev_list)

    return AnswerWithEvidence(answer=raw_response.strip(), evidence=[])
