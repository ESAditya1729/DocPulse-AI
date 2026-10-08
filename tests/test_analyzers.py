"""Unit tests for structured analyzers."""

import json

from docpulse.analyzers import (
    build_document_map_result,
    parse_answer_with_citations,
    parse_comparison_response,
    parse_concepts_response,
    parse_equations_response,
    parse_prerequisites_response,
    parse_study_response,
)
from docpulse.models import ParsedDocument, Section


def test_parse_concepts_valid_json():
    json_str = json.dumps(
        {
            "concepts": [
                {
                    "name": "Self-Attention",
                    "description": "Mechanism relating different positions of a single sequence.",
                    "importance": "high",
                    "sections": [2, 3],
                    "related_concepts": ["Multi-Head Attention", "Scaled Dot-Product"],
                    "prerequisites": ["Matrix Multiplication"],
                }
            ]
        }
    )
    res = parse_concepts_response(json_str, "test.pdf")
    assert res.document == "test.pdf"
    assert len(res.concepts) == 1
    c = res.concepts[0]
    assert c.name == "Self-Attention"
    assert c.importance == "high"
    assert c.sections == [2, 3]
    assert "Multi-Head Attention" in c.related_concepts


def test_parse_concepts_with_markdown_fences():
    text = """Here is the JSON:
```json
{
  "concepts": [
    {
      "name": "Positional Encoding",
      "description": "Injects order information",
      "importance": "medium"
    }
  ]
}
```
"""
    res = parse_concepts_response(text, "test.pdf")
    assert len(res.concepts) == 1
    assert res.concepts[0].name == "Positional Encoding"


def test_parse_concepts_malformed_fallback():
    malformed = """1. Transformer: An architecture based on attention mechanisms.
2. Residual Connection: Skip connection around sub-layers."""
    res = parse_concepts_response(malformed, "test.pdf")
    assert len(res.concepts) == 2
    assert res.concepts[0].name == "Transformer"
    assert "architecture based on attention" in res.concepts[0].description


def test_parse_prerequisites_valid():
    json_str = json.dumps(
        {
            "prerequisites": [
                {
                    "name": "Linear Algebra",
                    "importance": "high",
                    "difficulty": "medium",
                    "needed_for": "Matrix operations",
                    "relevant_sections": [1, 2],
                    "dependencies": ["Basic Algebra"],
                }
            ]
        }
    )
    res = parse_prerequisites_response(json_str, "paper.pdf")
    assert len(res.prerequisites) == 1
    p = res.prerequisites[0]
    assert p.name == "Linear Algebra"
    assert p.dependencies == ["Basic Algebra"]


def test_parse_equations_valid():
    json_str = json.dumps(
        {
            "equations": [
                {
                    "id": 1,
                    "latex": r"\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V",
                    "readable": "Attention(Q,K,V) = softmax(QKᵀ / √dₖ)V",
                    "section": "3.2",
                    "variables": [
                        {"symbol": "Q", "description": "Query matrix"},
                        {"symbol": "K", "description": "Key matrix"},
                    ],
                    "concepts": ["Self-Attention", "Softmax"],
                    "explanation": "Scaled dot product attention formula.",
                }
            ]
        }
    )
    res = parse_equations_response(json_str, "paper.pdf")
    assert len(res.equations) == 1
    eq = res.equations[0]
    assert eq.id == 1
    assert eq.readable == "Attention(Q,K,V) = softmax(QKᵀ / √dₖ)V"
    assert len(eq.variables) == 2
    assert eq.variables[0].symbol == "Q"


def test_parse_comparison():
    json_str = json.dumps(
        {
            "shared_concepts": ["Attention", "Transformer"],
            "unique_to_a": ["Absolute Positional Encoding"],
            "unique_to_b": ["Rotary Positional Embeddings"],
            "methodology_a": "Method A uses standard sinusoids.",
            "methodology_b": "Method B uses RoPE.",
            "key_differences": ["Relative vs Absolute positional encoding"],
            "conclusion": "Doc B improves length generalization.",
        }
    )
    res = parse_comparison_response(json_str, "doc_a.pdf", "doc_b.pdf")
    assert res.document_a == "doc_a.pdf"
    assert res.shared_concepts == ["Attention", "Transformer"]
    assert res.unique_to_b == ["Rotary Positional Embeddings"]
    assert "RoPE" in res.methodology_b


def test_parse_study_response():
    json_str = json.dumps(
        {
            "key_concepts": ["Self-Attention", "Feed-Forward"],
            "prerequisites": ["Matrix Multiplication"],
            "important_equations": ["Attention(Q,K,V)"],
            "flashcards": [{"question": "What is Q?", "answer": "Query matrix"}],
            "questions": [
                {
                    "question": "Why scale attention?",
                    "type": "conceptual",
                    "options": [],
                    "answer": "To prevent vanishing gradients in softmax",
                    "explanation": "Large dot products push softmax into regions with small gradients.",
                }
            ],
        }
    )
    res = parse_study_response(json_str, "doc.pdf")
    assert len(res.key_concepts) == 2
    assert len(res.flashcards) == 1
    assert res.flashcards[0].question == "What is Q?"
    assert len(res.questions) == 1
    assert res.questions[0].type == "conceptual"


def test_parse_study_response_with_null_options():
    json_str = json.dumps(
        {
            "key_concepts": ["Concept 1"],
            "prerequisites": None,
            "important_equations": None,
            "flashcards": [{"question": "Q1?", "answer": "A1"}],
            "questions": [
                {
                    "question": "What is linear attention?",
                    "type": "conceptual",
                    "options": None,
                    "answer": "Kernel approximation",
                    "explanation": None,
                }
            ],
        }
    )
    res = parse_study_response(json_str, "doc.pdf")
    assert len(res.questions) == 1
    assert res.questions[0].options == []
    assert res.questions[0].question == "What is linear attention?"


def test_build_document_map_and_mermaid():
    from docpulse.renderers import generate_document_map_mermaid

    doc = ParsedDocument(
        file_path="paper.pdf",  # type: ignore
        file_name="paper.pdf",
        doc_type="pdf",
        sections=[Section(id=1, title="Introduction", content="Intro text", level=1, page_number=1)],
        raw_text="Intro text",
    )
    concepts = parse_concepts_response(
        json.dumps({"concepts": [{"name": "Attention", "related_concepts": ["Softmax"]}]}), "paper.pdf"
    ).concepts
    prereqs = parse_prerequisites_response(
        json.dumps({"prerequisites": [{"name": "Linear Algebra"}]}), "paper.pdf"
    ).prerequisites
    equations = parse_equations_response(
        json.dumps({"equations": [{"id": 1, "readable": "A = B", "section": "1"}]}), "paper.pdf"
    ).equations

    doc_map = build_document_map_result(doc, concepts, prereqs, equations, ["Reference 1"])
    assert doc_map.document == "paper.pdf"
    assert len(doc_map.sections) == 1
    assert len(doc_map.core_concepts) == 1

    mermaid = generate_document_map_mermaid(doc_map)
    assert "graph TD" in mermaid
    assert "paper.pdf" in mermaid
    assert "Attention" in mermaid
    assert "Linear Algebra" in mermaid


def test_parse_answer_with_citations_json():
    json_str = json.dumps(
        {
            "answer": "Scaling prevents vanishing gradients in softmax.",
            "evidence": [
                {
                    "section": "3.2",
                    "section_title": "Attention",
                    "page": 4,
                    "excerpt": "We suspect that for large values of dk, the dot products grow large...",
                }
            ],
        }
    )
    res = parse_answer_with_citations(json_str)
    assert "Scaling prevents" in res.answer
    assert len(res.evidence) == 1
    assert res.evidence[0].section == "3.2"
    assert res.evidence[0].page == 4
