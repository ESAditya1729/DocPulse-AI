"""Prompts used across DocPulse commands."""

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
