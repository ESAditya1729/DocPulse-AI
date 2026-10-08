"""Renderers and formatters for human-readable, JSON, and Mermaid outputs."""

import json
import re
import sys
from typing import Any

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

from docpulse.mathbox import prettify_math, split_segments
from docpulse.models import (
    AnswerWithEvidence,
    ConceptIndexResult,
    DocumentComparisonResult,
    DocumentMapResult,
    EquationResult,
    PrerequisiteResult,
    StudyModeResult,
)

console = Console()


def render_markdown_with_equations(markdown_text: str) -> Markdown | Group:
    """Render LLM markdown, boxing each $...$ / $$...$$ equation in place."""
    segments = split_segments(markdown_text)
    if not any(kind == "math" for kind, _ in segments):
        return Markdown(markdown_text)

    parts = []
    for kind, chunk in segments:
        if kind == "math":
            parts.append(
                Panel(
                    Text(prettify_math(chunk), justify="center", style="bold white"),
                    border_style="magenta",
                    expand=False,
                    padding=(0, 2),
                )
            )
        elif chunk.strip():
            parts.append(Markdown(chunk))
    return Group(*parts)


def render_json(data: Any) -> str:
    """Standardized JSON string formatting for Pydantic models or dicts."""
    if hasattr(data, "model_dump"):
        return json.dumps(data.model_dump(), indent=2, ensure_ascii=False)
    return json.dumps(data, indent=2, ensure_ascii=False)


def _ensure_utf8(stream: Any) -> None:
    """Make a stream able to carry the unicode DocPulse deals in.

    On Windows, a redirected stdout is encoded with the legacy ANSI code page
    (cp1252 on most installs), so writing a single Sigma or integral sign raises
    UnicodeEncodeError. Papers are full of those, and JSON output has to stay
    machine-readable rather than abort mid-document.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    encoding = str(getattr(stream, "encoding", "") or "").lower().replace("-", "")
    if encoding.startswith("utf8"):
        return
    try:
        reconfigure(encoding="utf-8")
    except (ValueError, OSError, AttributeError):
        pass


def print_json(data: Any) -> None:
    """Write JSON straight to stdout, bypassing Rich.

    `console.print` is wrong for machine-readable output on two counts: it
    wraps long lines at the terminal width, injecting real newlines that turn
    the payload into invalid JSON, and it interprets `[...]` sequences in the
    data as console markup, so a document containing `[bold]` would silently
    lose that text. Anything aimed at a pipe or a JSON parser goes through
    here instead.
    """
    text = render_json(data)
    try:
        _ensure_utf8(sys.stdout)
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
    except UnicodeEncodeError:
        # Last resort: pure ASCII is still valid JSON, just less pretty.
        sys.stdout.write(json.dumps(data, indent=2, ensure_ascii=True) + "\n")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Concepts Renderer
# ---------------------------------------------------------------------------


def render_concepts_text(result: ConceptIndexResult) -> None:
    """Render structured concept index nicely in Rich terminal."""
    if not result.concepts:
        console.print("[yellow]No concepts identified in the document.[/yellow]")
        return

    console.print(f"[bold cyan]Concepts ({len(result.concepts)} identified)[/bold cyan]")
    console.print("[dim]" + "─" * 40 + "[/dim]")

    for idx, c in enumerate(result.concepts, start=1):
        importance_color = {
            "high": "bold red",
            "medium": "bold yellow",
            "low": "dim green",
        }.get(c.importance.lower(), "cyan")

        sections_str = ", ".join(str(s) for s in c.sections) if c.sections else "General"
        console.print(f"\n[bold]{idx}. {c.name}[/bold]")
        console.print(f"   [dim]Sections:[/dim] {sections_str}")
        console.print(f"   [dim]Importance:[/dim] [{importance_color}]{c.importance.capitalize()}[/{importance_color}]")
        if c.description:
            console.print(f"   [dim]Description:[/dim] {c.description}")
        if c.related_concepts:
            console.print(f"   [dim]Related:[/dim] {', '.join(c.related_concepts)}")
        if c.prerequisites:
            console.print(f"   [dim]Prerequisites:[/dim] {', '.join(c.prerequisites)}")


# ---------------------------------------------------------------------------
# Prerequisites Renderer
# ---------------------------------------------------------------------------


def render_prerequisites_text(result: PrerequisiteResult) -> None:
    """Render prerequisite analysis with dependency tree."""
    if not result.prerequisites:
        console.print("[yellow]No prerequisites identified in the document.[/yellow]")
        return

    console.print(f"[bold cyan]Prerequisites ({len(result.prerequisites)} required)[/bold cyan]")
    console.print("[dim]" + "─" * 40 + "[/dim]")

    for p in result.prerequisites:
        importance_color = {
            "high": "bold red",
            "medium": "bold yellow",
            "low": "dim green",
        }.get(p.importance.lower(), "cyan")

        diff_color = {
            "advanced": "red",
            "high": "red",
            "medium": "yellow",
            "beginner": "green",
            "low": "green",
        }.get(p.difficulty.lower(), "white")

        console.print(f"\n[bold]{p.name}[/bold]")
        console.print(f"  [dim]Importance:[/dim] [{importance_color}]{p.importance.capitalize()}[/{importance_color}]")
        console.print(f"  [dim]Difficulty:[/dim] [{diff_color}]{p.difficulty.capitalize()}[/{diff_color}]")
        if p.needed_for:
            console.print(f"  [dim]Needed for:[/dim] {p.needed_for}")
        if p.relevant_sections:
            console.print(f"  [dim]Relevant in:[/dim] {', '.join(str(s) for s in p.relevant_sections)}")
        if p.dependencies:
            console.print(f"  [dim]Dependencies:[/dim] {', '.join(p.dependencies)}")


# ---------------------------------------------------------------------------
# Equations Renderer
# ---------------------------------------------------------------------------


def render_equations_text(result: EquationResult) -> None:
    """Render mathematical equations with variables and explanations."""
    if not result.equations:
        console.print("[yellow]No mathematical equations identified in the document.[/yellow]")
        return

    console.print(f"[bold cyan]Equations ({len(result.equations)} identified)[/bold cyan]")
    console.print("[dim]" + "─" * 40 + "[/dim]")

    for eq in result.equations:
        console.print(f"\n[bold magenta]Equation {eq.id}[/bold magenta]")
        console.print("[dim]" + "─" * 20 + "[/dim]")

        # Equation box
        display_math = eq.readable or prettify_math(eq.latex) or eq.latex
        console.print(
            Panel(
                Text(display_math, justify="center", style="bold white"),
                border_style="magenta",
                expand=False,
                padding=(0, 2),
            )
        )

        if eq.section:
            console.print(f"[dim]Section:[/dim] {eq.section}")

        if eq.variables:
            console.print("\n[dim]Variables:[/dim]")
            for v in eq.variables:
                console.print(f"  [bold cyan]{v.symbol:6}[/bold cyan] {v.description}")

        if eq.concepts:
            console.print(f"\n[dim]Concepts:[/dim] {', '.join(eq.concepts)}")

        if eq.explanation:
            console.print(f"\n[dim]Explanation:[/dim] {eq.explanation}")

        if eq.usage:
            console.print(f"[dim]Usage:[/dim] {eq.usage}")


# ---------------------------------------------------------------------------
# Document Map Renderer & Mermaid Generator
# ---------------------------------------------------------------------------


def render_document_map_text(result: DocumentMapResult) -> None:
    """Render a rich tree representation of the document map."""
    tree = Tree(f"[bold cyan]DOCUMENT MAP: {result.document}[/bold cyan]")

    # Sections branch
    if result.sections:
        sec_branch = tree.add("[bold magenta]Sections[/bold magenta]")
        for sec in result.sections:
            loc = f" (Page {sec.page})" if sec.page else f" (H{sec.level})"
            sec_branch.add(f"[cyan][{sec.id}][/cyan] {sec.title}{loc}")

    # Prerequisites branch
    if result.prerequisites:
        prereq_branch = tree.add("[bold yellow]Prerequisites[/bold yellow]")
        for p in result.prerequisites:
            prereq_node = prereq_branch.add(f"[yellow]{p.name}[/yellow] [dim]({p.importance})[/dim]")
            if p.dependencies:
                for dep in p.dependencies:
                    prereq_node.add(f"[dim]Depends on: {dep}[/dim]")

    # Core Concepts branch
    if result.core_concepts:
        concept_branch = tree.add("[bold green]Core Concepts[/bold green]")
        for c in result.core_concepts:
            c_node = concept_branch.add(f"[green]{c.name}[/green] [dim]({c.importance})[/dim]")
            if c.related_concepts:
                for r in c.related_concepts:
                    c_node.add(f"[dim]Related: {r}[/dim]")

    # Equations branch
    if result.equations:
        eq_branch = tree.add("[bold magenta]Equations[/bold magenta]")
        for eq in result.equations:
            sec_ref = f" → Section {eq.section}" if eq.section else ""
            eq_branch.add(f"Eq. {eq.id}: [dim]{eq.readable or eq.latex}[/dim]{sec_ref}")

    # References branch
    if result.references:
        ref_branch = tree.add("[bold blue]References & Sources[/bold blue]")
        for ref in result.references:
            ref_branch.add(f"[dim]{ref}[/dim]")

    console.print(tree)


def _mermaid_id(prefix: str, raw: object, taken: set[str]) -> str:
    """Build a unique, syntactically valid Mermaid node ID for a label.

    Mermaid node IDs may only contain alphanumerics and underscores, so an
    equation keyed by a string such as "Eq (1) loss" would otherwise produce
    `Eq_Eq (1) loss[...]` and break the diagram. Collisions are resolved by
    appending a counter so repeated labels share one node instead of being
    emitted twice.
    """
    slug = re.sub(r"[^0-9A-Za-z]+", "_", str(raw)).strip("_")
    if not slug:
        slug = "node"
    candidate = f"{prefix}_{slug}" if prefix else slug
    if slug[0].isdigit():
        candidate = f"{candidate}_"
    if candidate in taken:
        suffix = 2
        while f"{candidate}_{suffix}" in taken:
            suffix += 1
        candidate = f"{candidate}_{suffix}"
    taken.add(candidate)
    return candidate


def _mermaid_label(text: str) -> str:
    """Escape a label for use inside a double-quoted Mermaid node."""
    cleaned = " ".join(str(text).split())
    return cleaned.replace('"', "'")


def generate_document_map_mermaid(result: DocumentMapResult) -> str:
    """Generate clean Mermaid diagram syntax representing the document map."""
    taken: set[str] = set()
    lines = ["graph TD", f'    Doc["📄 {_mermaid_label(result.document)}"]']

    # Sections subgraph
    if result.sections:
        lines.append("    subgraph Sections")
        for s in result.sections:
            node = _mermaid_id("Sec", s.id, taken)
            lines.append(f'        {node}["[{s.id}] {_mermaid_label(s.title)}"]')
        lines.append("    end")
        lines.append("    Doc --> Sections")

    # Prerequisites subgraph
    if result.prerequisites:
        lines.append("    subgraph Prerequisites")
        for p in result.prerequisites:
            node = _mermaid_id("Pre", p.name, taken)
            lines.append(f'        {node}["{_mermaid_label(p.name)}"]')
        lines.append("    end")
        lines.append("    Doc --> Prerequisites")

    # Concepts subgraph. Relations point at the existing concept node when the
    # name matches one, otherwise at a single shared node for that name - so a
    # concept mentioning "Softmax" twice doesn't declare two identical nodes.
    if result.core_concepts:
        lines.append("    subgraph CoreConcepts[Core Concepts]")
        concept_ids: dict[str, str] = {}
        for c in result.core_concepts:
            node = _mermaid_id("Concept", c.name, taken)
            concept_ids[c.name.strip().lower()] = node
            lines.append(f'        {node}["{_mermaid_label(c.name)}"]')

        relation_ids: dict[str, str] = {}
        for c in result.core_concepts:
            source = concept_ids[c.name.strip().lower()]
            for rel in c.related_concepts:
                key = rel.strip().lower()
                if not key:
                    continue
                if key in concept_ids:
                    lines.append(f"        {source} -.-> {concept_ids[key]}")
                    continue
                target = relation_ids.get(key)
                if target is None:
                    target = _mermaid_id("Rel", rel, taken)
                    relation_ids[key] = target
                    lines.append(f'        {target}["{_mermaid_label(rel)}"]')
                lines.append(f"        {source} -.-> {target}")
        lines.append("    end")
        lines.append("    Doc --> CoreConcepts")

    # Equations subgraph
    if result.equations:
        lines.append("    subgraph Equations")
        for eq in result.equations:
            node = _mermaid_id("Eq", eq.id, taken)
            label = f"Eq {eq.id}: {eq.readable or 'Formula'}"
            lines.append(f'        {node}["{_mermaid_label(label)}"]')
        lines.append("    end")
        lines.append("    Doc --> Equations")

    # References branch
    if result.references:
        lines.append("    subgraph References")
        for idx, ref in enumerate(result.references):
            lines.append(f'        Ref_{idx}["{_mermaid_label(ref)}"]')
        lines.append("    end")
        lines.append("    Doc --> References")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Document Comparison Renderer
# ---------------------------------------------------------------------------


def render_comparison_text(result: DocumentComparisonResult) -> None:
    """Render comparison between two documents."""
    console.print("[bold cyan]DOCUMENT COMPARISON[/bold cyan]")
    console.print(f"[dim]Document A:[/dim] [bold]{result.document_a}[/bold]")
    console.print(f"[dim]Document B:[/dim] [bold]{result.document_b}[/bold]")
    console.print("[dim]" + "═" * 45 + "[/dim]\n")

    if result.shared_concepts:
        console.print("[bold green]Shared Concepts[/bold green]")
        for sc in result.shared_concepts:
            console.print(f"  • {sc}")
        console.print()

    if result.unique_to_a:
        console.print(f"[bold cyan]Unique to {result.document_a}[/bold cyan]")
        for u in result.unique_to_a:
            console.print(f"  • {u}")
        console.print()

    if result.unique_to_b:
        console.print(f"[bold magenta]Unique to {result.document_b}[/bold magenta]")
        for u in result.unique_to_b:
            console.print(f"  • {u}")
        console.print()

    if result.methodology_a or result.methodology_b:
        console.print("[bold yellow]Methodology[/bold yellow]")
        if result.methodology_a:
            console.print(f"  [cyan]{result.document_a}:[/cyan] {result.methodology_a}")
        if result.methodology_b:
            console.print(f"  [magenta]{result.document_b}:[/magenta] {result.methodology_b}")
        console.print()

    if result.prerequisites_comparison:
        console.print("[bold yellow]Prerequisites Comparison[/bold yellow]")
        console.print(f"  {result.prerequisites_comparison}\n")

    if result.equations_comparison:
        console.print("[bold yellow]Mathematical Formulations Comparison[/bold yellow]")
        console.print(f"  {result.equations_comparison}\n")

    if result.key_differences:
        console.print("[bold red]Key Differences[/bold red]")
        for diff in result.key_differences:
            console.print(f"  • {diff}")
        console.print()

    if result.conclusion:
        console.print(
            Panel(
                Markdown(result.conclusion),
                title="Comparative Conclusion",
                border_style="green",
                expand=False,
            )
        )


# ---------------------------------------------------------------------------
# Study Mode Renderer
# ---------------------------------------------------------------------------


def render_study_mode_text(result: StudyModeResult) -> None:
    """Render study mode questions, concepts, and flashcards."""
    console.print(f"[bold cyan]STUDY MODE: {result.document}[/bold cyan]")
    console.print("[dim]" + "═" * 45 + "[/dim]\n")

    if result.key_concepts:
        console.print("[bold green]Key Concepts[/bold green]")
        for idx, kc in enumerate(result.key_concepts, 1):
            console.print(f"  {idx}. {kc}")
        console.print()

    if result.prerequisites:
        console.print("[bold yellow]Prerequisites to Review[/bold yellow]")
        for _idx, pr in enumerate(result.prerequisites, 1):
            console.print(f"  • {pr}")
        console.print()

    if result.important_equations:
        console.print("[bold magenta]Important Equations & Formulas[/bold magenta]")
        for eq in result.important_equations:
            console.print(f"  • {eq}")
        console.print()

    if result.flashcards:
        console.print("[bold cyan]Flashcards[/bold cyan]")
        console.print("[dim]" + "─" * 20 + "[/dim]")
        for idx, fc in enumerate(result.flashcards, 1):
            console.print(
                Panel(
                    f"[bold yellow]Q:[/bold yellow] {fc.question}\n\n[bold green]A:[/bold green] {fc.answer}",
                    title=f"Card {idx}",
                    border_style="cyan",
                    expand=False,
                )
            )

    if result.questions:
        console.print("\n[bold cyan]Questions & Self-Assessment[/bold cyan]")
        console.print("[dim]" + "─" * 20 + "[/dim]")
        for idx, q in enumerate(result.questions, 1):
            type_tag = f" [dim]({q.type})[/dim]" if q.type else ""
            console.print(f"\n[bold]{idx}. {q.question}[/bold]{type_tag}")
            if q.options:
                for opt in q.options:
                    console.print(f"   {opt}")
            if q.answer:
                console.print(f"   [dim green]Answer:[/dim green] {q.answer}")
            if q.explanation:
                console.print(f"   [dim]Explanation:[/dim] {q.explanation}")


# ---------------------------------------------------------------------------
# Answer with Citations Renderer
# ---------------------------------------------------------------------------


def render_answer_with_citations_text(result: AnswerWithEvidence) -> None:
    """Render question answer along with grounded evidence citations."""
    console.print(
        Panel(
            render_markdown_with_equations(result.answer),
            title="Answer",
            border_style="blue",
            expand=False,
        )
    )

    if result.evidence:
        console.print("\n[bold cyan]Evidence & Citations[/bold cyan]")
        console.print("[dim]" + "─" * 20 + "[/dim]")
        for ev in result.evidence:
            locs = []
            if ev.section:
                locs.append(f"Section: {ev.section}")
            if ev.section_title:
                locs.append(f"Title: {ev.section_title}")
            if ev.page:
                locs.append(f"Page: {ev.page}")

            loc_str = " | ".join(locs) if locs else "Document Location"
            console.print(f"• [bold]{loc_str}[/bold]")
            if ev.excerpt:
                console.print(f"  [dim]\"{ev.excerpt}\"[/dim]")
