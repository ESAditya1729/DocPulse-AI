"""Prompts used across DocPulse commands."""

JSON_SYSTEM_PROMPT = """You are DocPulse, a document intelligence engine.
Output valid JSON only. Do not include markdown code fences, reasoning preambles, or conversational filler.
Start directly with `{` and end with `}`."""

SYSTEM_PROMPT = """You are DocPulse, a document intelligence engine.
Output ONLY the requested Markdown document directly.
Do NOT include conversational filler, meta-announcements, reasoning preambles (e.g. "Sure!", "Here is the study guide...", "Let me structure this..."), or closing pleasantries.
Start immediately with the top-level Markdown header.

Mathematical notation rules (output is rendered in a plain terminal and in plain Markdown, with no LaTeX/KaTeX support):
- Prefer plain Unicode symbols over LaTeX commands wherever an equation can be flattened to them
  (e.g. write γ, θ, Σ, ×, ², ₙ, † instead of \\gamma, \\theta, \\sum, \\times, ^2, _n, \\dagger).
- Only fall back to LaTeX for equations that genuinely cannot be flattened to Unicode (matrices,
  multi-line derivations), and in that case use ONLY dollar delimiters: $...$ inline, $$...$$ on
  their own line for display equations.
- NEVER use \\( \\), \\[ \\], or bare parentheses/brackets as math delimiters. Markdown's backslash-escaping
  strips the backslash from those (e.g. "\\( \\gamma \\)" renders as "( \\gamma )"), making the equation
  unreadable."""

ANALYZE_PROMPT = """Analyze the following document content and provide a structured breakdown.

Document Content:
\"\"\"{content}\"\"\"

Output format (Markdown):
# Executive Summary
(2-3 concise paragraphs)

# Core Themes & Topics
(Bullet points with clear explanations)

# Prerequisites & Assumed Knowledge
(What foundational background is needed?)

# Key Takeaways & Implications
(Direct practical takeaways)
"""

SECTION_PROMPT = """Provide an in-depth, structured analysis of the following document section.

Section Title: {title}
Section Content:
\"\"\"{content}\"\"\"

Output format (Markdown):
# Summary
(Comprehensive summary of this section)

# Critical Insights & Core Points
(Key mechanisms, equations, arguments, or architectures)

# Potential Pitfalls & Edge Cases
(Common misunderstandings or edge cases)

# Review Questions
(2-3 comprehension test questions with concise answers)
"""

SOURCES_PROMPT = """Extract all external references, cited academic papers, GitHub repositories, datasets, tools, and URLs from the following document content.

Document Content:
\"\"\"{content}\"\"\"

Output format (Markdown):
# Cited Papers & Academic Literature
(List citations or write 'None explicitly cited')

# Code Repositories & Libraries
(List repositories/libraries or write 'None explicitly cited')

# External URLs & Documentation
(List URLs or write 'None explicitly cited')

# Datasets & Benchmarks
(List datasets or write 'None explicitly cited')
"""

MAP_PROMPT = """You are condensing part {index} of {total} of a larger document into dense notes for later synthesis.
Preserve concrete facts, definitions, equations (verbatim), names, numbers, and arguments. Do not add commentary,
do not omit specifics for brevity, and do not try to summarize the whole document - just this excerpt.
Output plain condensed notes only, no headers, no meta-commentary.

Excerpt:
\"\"\"{content}\"\"\"
"""

ASK_FIRST_PROMPT = """You will answer follow-up questions about the document below across multiple turns.
Read it carefully, then answer ONLY the question at the end. Keep answers focused, cite section titles or
specific details when relevant, and say clearly if the document doesn't contain the answer.

Document: {doc_name}
Document Content:
\"\"\"{content}\"\"\"

Question: {question}
"""

EXPORT_PROMPT = """Generate a comprehensive Study Guide / README in Markdown based on the provided document content.

Document Title: {doc_name}
Document Content:
\"\"\"{content}\"\"\"

You MUST follow this exact structure without any conversational chatter or introductory preamble:

# {doc_name} — Study Guide & Notes

## Overview & Objectives
(High-level summary and learning objectives)

## Chapter & Section Breakdown
(Deep-dive breakdown with key explanations and code/concept snippets)

## Key Definitions & Glossary
(Terms, acronyms, and operational definitions)

## Practical Exercises & Experiments
(Actionable exercises or experiments to reinforce understanding)

## Self-Assessment Quiz
(Questions followed by an Answers & Explanations section)
"""

CONCEPTS_PROMPT = """Extract the key concepts from the entire document content.
Output a JSON object with this exact structure:
{{
  "concepts": [
    {{
      "name": "Concept Name",
      "description": "Clear and concise explanation of the concept based on the document",
      "importance": "high | medium | low",
      "sections": ["Section 1 or ID", "Section 2"],
      "related_concepts": ["Other Concept 1", "Other Concept 2"],
      "prerequisites": ["Prerequisite Concept 1"]
    }}
  ]
}}

Document Content:
\"\"\"{content}\"\"\"
"""

PREREQUISITES_PROMPT = """Analyze the prerequisite knowledge and background needed to understand the document.
Output a JSON object with this exact structure:
{{
  "prerequisites": [
    {{
      "name": "Prerequisite Topic / Subject",
      "importance": "high | medium | low",
      "difficulty": "beginner | medium | advanced",
      "needed_for": "Why this is needed and what it enables in this document",
      "relevant_sections": ["Section 1", "Section 2"],
      "dependencies": ["Underlying foundational topic (e.g. Linear Algebra for Matrix Ops)"]
    }}
  ]
}}

Document Content:
\"\"\"{content}\"\"\"
"""

EQUATIONS_PROMPT = """Identify and explain all mathematical equations, formulations, and algorithms in the document.
Output a JSON object with this exact structure:
{{
  "equations": [
    {{
      "id": 1,
      "latex": "LaTeX or raw representation of equation",
      "readable": "Unicode/human-friendly representation",
      "section": "Section number or title where it appears",
      "variables": [
        {{"symbol": "x", "description": "variable description"}}
      ],
      "concepts": ["Concept 1", "Concept 2"],
      "explanation": "Clear explanation of what the equation computes and its significance",
      "usage": "Where and how this equation is used in the methodology or architecture"
    }}
  ]
}}

If no equations exist in the document, return {{"equations": []}}.

Document Content:
\"\"\"{content}\"\"\"
"""

COMPARE_PROMPT = """Compare the following two documents conceptually and structurally.
Document A ({doc_a_name}):
\"\"\"{doc_a_content}\"\"\"

Document B ({doc_b_name}):
\"\"\"{doc_b_content}\"\"\"

Output a JSON object with this exact structure:
{{
  "shared_concepts": ["Concept present in both"],
  "unique_to_a": ["Concept or method unique to Document A"],
  "unique_to_b": ["Concept or method unique to Document B"],
  "methodology_a": "Summary of Document A's methodology and technical approach",
  "methodology_b": "Summary of Document B's methodology and technical approach",
  "prerequisites_comparison": "Comparison of background and prerequisite knowledge needed",
  "equations_comparison": "Comparison of mathematical foundations and formulations",
  "key_differences": [
    "Key difference 1",
    "Key difference 2"
  ],
  "conclusion": "Synthesized conclusion comparing their contributions, strengths, and use cases"
}}
"""

STUDY_PROMPT = """Generate an interactive learning and study package based on the document content.
Create {num_questions} questions across multiple-choice, conceptual, and short-answer types, along with flashcards.

Output a JSON object with this exact structure:
{{
  "key_concepts": ["Concept 1", "Concept 2"],
  "prerequisites": ["Prerequisite 1", "Prerequisite 2"],
  "important_equations": ["Equation 1 or formula"],
  "flashcards": [
    {{
      "question": "Concise prompt/question for active recall?",
      "answer": "Direct, precise answer"
    }}
  ],
  "questions": [
    {{
      "question": "Question text?",
      "type": "multiple_choice | conceptual | short_answer",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "answer": "Correct answer or answer key",
      "explanation": "Detailed explanation of why this is correct"
    }}
  ]
}}

Document Content:
\"\"\"{content}\"\"\"
"""

ASK_CITATIONS_PROMPT = """Answer the user's question based on the document below.
Provide a clear answer along with citations indicating the section, page (if available), and relevant excerpt where the evidence is found.

Output a JSON object with this exact structure:
{{
  "answer": "Comprehensive and accurate answer to the question",
  "evidence": [
    {{
      "section": "Section number or identifier",
      "section_title": "Section title",
      "page": null,
      "excerpt": "Brief excerpt or summary of evidence from that section"
    }}
  ]
}}

Document: {doc_name}
Document Content:
\"\"\"{content}\"\"\"

Question: {question}
"""
