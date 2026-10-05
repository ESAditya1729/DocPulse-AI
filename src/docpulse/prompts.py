"""Prompts used across DocPulse commands."""

SYSTEM_PROMPT = """You are DocPulse, a document intelligence engine.
Output ONLY the requested Markdown document directly.
Do NOT include conversational filler, meta-announcements, reasoning preambles (e.g. "Sure!", "Here is the study guide...", "Let me structure this..."), or closing pleasantries.
Start immediately with the top-level Markdown header."""

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
