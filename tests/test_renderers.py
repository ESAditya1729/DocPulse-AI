"""Tests for output rendering, especially Mermaid diagram validity."""

import json
import re
import sys

from docpulse.models import (
    ConceptItem,
    DocumentMapResult,
    DocumentMapSection,
    EquationItem,
    PrerequisiteItem,
)
from docpulse.renderers import _ensure_utf8, generate_document_map_mermaid, print_json, render_json

# Mermaid accepts only these characters in a node identifier.
VALID_NODE_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
NODE_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[|\.->|-->)")


def node_ids(diagram: str) -> list[str]:
    """Every declared or referenced node identifier in a Mermaid diagram."""
    found: list[str] = []
    for line in diagram.splitlines():
        stripped = line.strip()
        if stripped.startswith("subgraph") or stripped in ("graph TD", "end"):
            continue
        match = NODE_LINE.match(stripped)
        if match:
            found.append(match.group(1))
    return found


def build_map(**overrides) -> DocumentMapResult:
    defaults = {
        "document": "paper.pdf",
        "sections": [DocumentMapSection(id=1, title="Introduction", level=1, page=1)],
        "core_concepts": [],
        "prerequisites": [],
        "equations": [],
        "references": [],
    }
    defaults.update(overrides)
    return DocumentMapResult(**defaults)


# --- Mermaid validity -------------------------------------------------------


def test_every_node_id_is_syntactically_valid():
    result = build_map(
        core_concepts=[
            ConceptItem(name="Attention", related_concepts=["Softmax"]),
            ConceptItem(name="Softmax"),
        ],
        prerequisites=[PrerequisiteItem(name="Linear Algebra", dependencies=["Calculus"])],
        equations=[EquationItem(id="Eq (1) loss", readable="L = -Σ y log p")],
    )

    diagram = generate_document_map_mermaid(result)

    for node in node_ids(diagram):
        assert VALID_NODE_ID.match(node), f"invalid Mermaid node id: {node!r}"


def test_string_equation_ids_produce_sanitized_nodes():
    diagram = generate_document_map_mermaid(build_map(equations=[EquationItem(id="Eq (1) loss", readable="L = x")]))

    assert "Eq_Eq_1_loss" in diagram
    assert "Eq (1) loss[" not in diagram


def test_duplicate_relations_emit_a_single_shared_node():
    """Two mentions of the same relation must not declare two identical nodes."""
    result = build_map(
        core_concepts=[
            ConceptItem(name="Attention", related_concepts=["Softmax", "Softmax"]),
            ConceptItem(name="Multi-Head", related_concepts=["Softmax"]),
        ]
    )

    diagram = generate_document_map_mermaid(result)
    declarations = [line for line in diagram.splitlines() if line.strip().startswith("Rel_") and "[" in line]

    assert len(declarations) == 1
    # One declaration plus three edges (Attention names Softmax twice).
    assert diagram.count("Rel_Softmax") == 4


def test_relation_matching_an_existing_concept_links_to_that_concept():
    result = build_map(
        core_concepts=[
            ConceptItem(name="Attention", related_concepts=["Softmax"]),
            ConceptItem(name="Softmax"),
        ]
    )

    diagram = generate_document_map_mermaid(result)

    assert "Rel_Softmax" not in diagram
    assert diagram.count("Concept_Softmax") == 2, "declaration plus the relation edge"


def test_repeated_concept_names_get_distinct_nodes():
    diagram = generate_document_map_mermaid(
        build_map(core_concepts=[ConceptItem(name="Attention"), ConceptItem(name="Attention")])
    )

    assert "Concept_Attention" in diagram
    assert "Concept_Attention_2" in diagram


def test_double_quotes_in_labels_are_escaped():
    diagram = generate_document_map_mermaid(build_map(core_concepts=[ConceptItem(name='the "core" idea')]))

    assert 'the "core" idea' not in diagram
    assert "the 'core' idea" in diagram


def test_references_are_included_as_a_subgraph():
    diagram = generate_document_map_mermaid(build_map(references=["Attention Is All You Need", "arXiv:1706.03762"]))

    assert "subgraph References" in diagram
    assert "Attention Is All You Need" in diagram
    assert "Doc --> References" in diagram


def test_empty_map_still_produces_a_valid_diagram():
    diagram = generate_document_map_mermaid(build_map())

    assert diagram.startswith("graph TD")
    assert "paper.pdf" in diagram


def test_node_ids_are_unique_across_the_whole_diagram():
    result = build_map(
        core_concepts=[ConceptItem(name="Attention")],
        prerequisites=[PrerequisiteItem(name="Attention")],
        equations=[EquationItem(id="Attention", readable="x")],
    )

    ids = node_ids(generate_document_map_mermaid(result))
    prefixes = [i.rsplit("_", 1)[0] for i in ids if not i.startswith("Doc")]

    assert len(prefixes) == len(set(prefixes)), "distinct subgraphs must not collide on one id"


# --- JSON rendering ---------------------------------------------------------


def test_render_json_handles_pydantic_models_and_dicts():
    result = build_map(core_concepts=[ConceptItem(name="Attention")])

    assert '"document": "paper.pdf"' in render_json(result)
    assert render_json({"a": 1}) == '{\n  "a": 1\n}'


def test_render_json_keeps_unicode_readable():
    assert "Σ" in render_json({"eq": "Σ x"})


# --- Machine-readable output must survive a pipe ---------------------------


def test_print_json_emits_exactly_one_line_per_field(capsys):
    """Rich would wrap long values at the terminal width and corrupt the JSON."""
    print_json({"body": "x" * 500})

    out = capsys.readouterr().out
    assert json.loads(out)["body"] == "x" * 500
    # Three newlines: after '{', after the single field, after '}'. A wrapped
    # value would add dozens more.
    assert out.count("\n") == 3


def test_print_json_does_not_consume_bracket_markup(capsys):
    """`[bold]` in a document must not be parsed as console markup and vanish."""
    print_json({"content": "A [bold] claim [/bold] and [red]more[/red]"})

    assert json.loads(capsys.readouterr().out)["content"] == "A [bold] claim [/bold] and [red]more[/red]"


def test_print_json_round_trips_unicode(capsys):
    value = "\u03a3 \u03bb \u222b \u221a2 \u2014 em-dash"
    print_json({"eq": value})

    assert json.loads(capsys.readouterr().out)["eq"] == value


def test_print_json_falls_back_to_ascii_when_the_stream_cannot_encode(monkeypatch):
    class PickyStream:
        """Rejects non-ASCII and, like a captured stdout, cannot be reconfigured."""

        encoding = "ascii"
        errors = "strict"

        def __init__(self):
            self.chunks = []

        def write(self, text):
            self.chunks.append(text.encode("ascii", errors="strict").decode("ascii"))

        def flush(self):
            pass

    stream = PickyStream()
    monkeypatch.setattr(sys, "stdout", stream)

    print_json({"eq": "\u03a3"})

    # ensure_ascii=True still parses back to the original character.
    assert json.loads(stream.chunks[0])["eq"] == "\u03a3"


def test_ensure_utf8_upgrades_a_legacy_codepage_stream():
    class LegacyStream:
        def __init__(self):
            self.reconfigured_to = None

        def reconfigure(self, encoding):
            self.reconfigured_to = encoding

    stream = LegacyStream()
    stream.encoding = "cp1252"

    _ensure_utf8(stream)

    assert stream.reconfigured_to == "utf-8"


def test_ensure_utf8_leaves_utf8_streams_alone():
    class AlreadyUtf8:
        def __init__(self):
            self.reconfigured_to = None
            self.encoding = "utf-8"

        def reconfigure(self, encoding):
            self.reconfigured_to = encoding

    stream = AlreadyUtf8()
    _ensure_utf8(stream)

    assert stream.reconfigured_to is None