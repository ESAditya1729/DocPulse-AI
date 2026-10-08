"""Lightweight local retrieval for grounding citations in source text.

DocPulse's `ask --citations` used to trust the LLM's claims about where
evidence came from. This module indexes the parsed document with BM25 (a
classic sparse retrieval ranking - no embeddings, no vector store, no extra
dependencies) so that:

1. the prompt can show the model the passages most relevant to the question,
   and
2. each returned citation can be verified against the text it claims to
   quote, with a retrieval score attached.

Everything here is deterministic and runs in-process on the parsed document.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

from docpulse.models import CitationEvidence, Section

# Passage sizing for BM25 indexing. Sections larger than this are split on
# line/sentence boundaries with a small overlap so a claim straddling a split
# still matches one of the two passages.
PASSAGE_MAX_CHARS = 1200
PASSAGE_OVERLAP_CHARS = 200

# An excerpt counts as verified when this fraction of its content tokens
# appears in the section it cites. LLM excerpts are near-verbatim quotes, so
# a generous threshold avoids flagging paraphrases as false while still
# catching section numbers the model invented.
VERIFICATION_THRESHOLD = 0.5

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens - the unit BM25 and verification share."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class Passage:
    """A retrievable chunk of one document section."""

    section_id: int
    section_title: str
    page: int | None
    text: str


def build_passages(sections: list[Section]) -> list[Passage]:
    """Split sections into indexable passages, oversized ones with overlap."""
    passages: list[Passage] = []
    for sec in sections:
        text = (sec.content or sec.title).strip()
        if not text:
            continue
        if len(text) <= PASSAGE_MAX_CHARS:
            passages.append(Passage(sec.id, sec.title, sec.page_number, text))
            continue

        start = 0
        length = len(text)
        while start < length:
            end = min(start + PASSAGE_MAX_CHARS, length)
            if end < length:
                window = text[start:end]
                boundary = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(". "))
                if boundary > PASSAGE_MAX_CHARS // 2:
                    end = start + boundary + 1
            chunk = text[start:end].strip()
            if chunk:
                passages.append(Passage(sec.id, sec.title, sec.page_number, chunk))
            if end >= length:
                break
            start = max(end - PASSAGE_OVERLAP_CHARS, start + 1)
    return passages


class BM25Index:
    """Okapi BM25 over document passages."""

    def __init__(self, passages: list[Passage], *, k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.k1 = k1
        self.b = b
        self._term_freqs: list[Counter[str]] = [Counter(tokenize(p.text)) for p in passages]
        self._doc_lens = [sum(tf.values()) for tf in self._term_freqs]
        self._avg_dl = (sum(self._doc_lens) / len(self._doc_lens)) if self._doc_lens else 0.0
        self._doc_freq: Counter[str] = Counter()
        for tf in self._term_freqs:
            for term in tf:
                self._doc_freq[term] += 1

    def __len__(self) -> int:
        return len(self.passages)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Passage, float]]:
        """Return the passages ranked most relevant to the query, best first."""
        if not self.passages or self._avg_dl == 0.0:
            return []
        total = len(self.passages)
        scores = [0.0] * total
        for term in set(tokenize(query)):
            df = self._doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self._term_freqs):
                freq = tf.get(term, 0)
                if not freq:
                    continue
                norm = self.k1 * (1.0 - self.b + self.b * self._doc_lens[i] / self._avg_dl)
                scores[i] += idf * freq * (self.k1 + 1.0) / (freq + norm)

        ranked = sorted(range(total), key=lambda i: scores[i], reverse=True)
        return [(self.passages[i], scores[i]) for i in ranked[:top_k] if scores[i] > 0.0]


def render_passages(hits: list[tuple[Passage, float]]) -> str:
    """Format retrieval hits for injection into the citations prompt."""
    if not hits:
        return "(no matching passages retrieved - rely on the document content below)"
    lines = []
    for idx, (passage, score) in enumerate(hits, start=1):
        location = f"Section {passage.section_id}: {passage.section_title}"
        if passage.page:
            location += f" (page {passage.page})"
        lines.append(f"{idx}. [{location}] (relevance {score:.2f})")
        lines.append(f"   {passage.text[:400]}")
    return "\n".join(lines)


def section_scores(index: BM25Index, query: str) -> dict[int, float]:
    """Best BM25 score achieved by each section for the query."""
    scores: dict[int, float] = {}
    for passage, score in index.search(query, top_k=len(index)):
        current = scores.get(passage.section_id)
        if current is None or score > current:
            scores[passage.section_id] = score
    return scores


def resolve_section(evidence: CitationEvidence, sections: list[Section]) -> Section | None:
    """Find the section an evidence item claims to come from, if any."""
    if evidence.section is not None:
        raw = str(evidence.section).strip()
        if raw.isdigit():
            for sec in sections:
                if sec.id == int(raw):
                    return sec
        for sec in sections:
            if sec.title.strip().lower() == raw.lower():
                return sec
    if evidence.section_title:
        claimed = evidence.section_title.strip().lower()
        for sec in sections:
            if sec.title.strip().lower() == claimed:
                return sec
        for sec in sections:
            if claimed in sec.title.strip().lower() or sec.title.strip().lower() in claimed:
                return sec
    return None


def containment_ratio(excerpt: str, text: str) -> float:
    """Fraction of excerpt tokens present in the source text."""
    excerpt_tokens = tokenize(excerpt)
    if not excerpt_tokens:
        return 0.0
    source_tokens = set(tokenize(text))
    if not source_tokens:
        return 0.0
    hits = sum(1 for token in excerpt_tokens if token in source_tokens)
    return hits / len(excerpt_tokens)


def verify_evidence(
    evidence: list[CitationEvidence],
    sections: list[Section],
    scores: dict[int, float],
) -> None:
    """Attach `verified` and `score` to each evidence item, in place.

    `verified` is True when the claimed section resolves and the excerpt
    actually appears in it, False when it doesn't, and None when there is no
    excerpt to check. `score` is the section's BM25 score for the question,
    or None when the section could not be resolved.
    """
    for ev in evidence:
        section = resolve_section(ev, sections)
        ev.score = scores.get(section.id) if section is not None else None
        if ev.excerpt and section is not None:
            ev.verified = containment_ratio(ev.excerpt, section.content or section.title) >= VERIFICATION_THRESHOLD
        elif ev.excerpt and section is None:
            ev.verified = False
        else:
            ev.verified = None
