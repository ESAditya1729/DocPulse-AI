"""Unit tests for BM25 retrieval and citation verification."""

from docpulse.models import CitationEvidence, Section
from docpulse.retrieval import (
    PASSAGE_MAX_CHARS,
    BM25Index,
    build_passages,
    containment_ratio,
    render_passages,
    resolve_section,
    section_scores,
    tokenize,
    verify_evidence,
)


def _section(sec_id: int, title: str, content: str) -> Section:
    return Section(id=sec_id, title=title, content=content, level=1)


def test_tokenize_lowercases_and_splits():
    assert tokenize("Hello, World! x2") == ["hello", "world", "x2"]
    assert tokenize("") == []


def test_build_passages_keeps_small_section_whole():
    sec = _section(1, "Intro", "Short section text.")
    passages = build_passages([sec])
    assert len(passages) == 1
    assert passages[0].section_id == 1
    assert passages[0].text == "Short section text."


def test_build_passages_indexes_section_by_title_when_body_empty():
    # A section with no body is still indexed via its title so a question
    # about that heading can still find it.
    passages = build_passages([_section(1, "Empty", "")])
    assert len(passages) == 1
    assert passages[0].text == "Empty"


def test_build_passages_splits_oversized_section_and_covers_all_text():
    paragraphs = [f"Paragraph {i} covers unique topic words number {i} in detail." for i in range(30)]
    sec = _section(7, "Long", "\n\n".join(paragraphs))
    passages = build_passages([sec])

    assert len(passages) > 1
    assert all(p.section_id == 7 for p in passages)
    assert all(len(p.text) <= PASSAGE_MAX_CHARS + 1 for p in passages)
    # First and last paragraphs both survive the split.
    covered = " ".join(p.text for p in passages)
    assert "unique topic words number 0" in covered
    assert "unique topic words number 29" in covered


def test_bm25_ranks_relevant_passage_first():
    sections = [
        _section(1, "Cooking", "Boil water, add pasta, stir occasionally and season with salt."),
        _section(2, "Physics", "Quantum entanglement links particles so measurements stay correlated."),
        _section(3, "Gardening", "Water the plants daily and ensure the soil drains properly."),
    ]
    index = BM25Index(build_passages(sections))

    hits = index.search("quantum entanglement particles", top_k=3)
    assert hits
    assert hits[0][0].section_id == 2
    assert hits[0][1] > 0.0


def test_bm25_respects_top_k_and_drops_zero_scores():
    sections = [_section(i, f"S{i}", f"distinct content block number {i}") for i in range(1, 6)]
    index = BM25Index(build_passages(sections))

    assert len(index.search("content number", top_k=2)) == 2
    assert index.search("unrelatedtermnotinthedoc") == []


def test_bm25_empty_index_is_safe():
    index = BM25Index([])
    assert len(index) == 0
    assert index.search("anything") == []


def test_render_passages_formats_hits_and_handles_empty():
    sec = _section(1, "Intro", "Some intro text.")
    index = BM25Index(build_passages([sec]))
    hits = index.search("intro text")

    rendered = render_passages(hits)
    assert "Section 1: Intro" in rendered
    assert "relevance" in rendered

    fallback = render_passages([])
    assert "no matching passages" in fallback


def test_section_scores_keeps_best_score_per_section():
    long_section = _section(1, "A", "attention attention attention details")
    other = _section(2, "B", "unrelated gardening advice")
    index = BM25Index(build_passages([long_section, other]))

    scores = section_scores(index, "attention")
    assert set(scores) == {1}
    assert scores[1] > 0.0


def test_resolve_section_by_id_title_and_fallback():
    sections = [_section(1, "Introduction", "text"), _section(2, "Methodology", "text")]

    assert resolve_section(CitationEvidence(section=1), sections).id == 1
    assert resolve_section(CitationEvidence(section="2"), sections).id == 2
    assert resolve_section(CitationEvidence(section="Methodology"), sections).id == 2
    assert resolve_section(CitationEvidence(section_title="methodology"), sections).id == 2
    assert resolve_section(CitationEvidence(section=99), sections) is None
    assert resolve_section(CitationEvidence(), sections) is None


def test_containment_ratio():
    text = "the transformer uses scaled dot product attention"
    assert containment_ratio("scaled dot product attention", text) == 1.0
    assert containment_ratio("", text) == 0.0
    assert containment_ratio("attention", text) == 1.0
    assert containment_ratio("completely different words", text) == 0.0


def test_verify_evidence_marks_verified_false_and_unchecked():
    sections = [_section(1, "Attention", "The transformer uses scaled dot product attention.")]
    scores = {1: 4.2}

    verified = CitationEvidence(section=1, excerpt="scaled dot product attention")
    wrong_place = CitationEvidence(section=99, excerpt="a quote about penguins in Antarctica")
    no_excerpt = CitationEvidence(section=1)

    verify_evidence([verified, wrong_place, no_excerpt], sections, scores)

    assert verified.verified is True
    assert verified.score == 4.2
    assert wrong_place.verified is False
    assert wrong_place.score is None
    assert no_excerpt.verified is None
    assert no_excerpt.score == 4.2


def test_verify_evidence_flags_invented_excerpt_in_real_section():
    sections = [_section(1, "Attention", "The transformer uses scaled dot product attention.")]
    hallucinated = CitationEvidence(section=1, excerpt="the model never mentions this sentence at all")

    verify_evidence([hallucinated], sections, {})

    assert hallucinated.verified is False
