"""Renderers and formatters for human-readable, JSON, and Mermaid outputs."""

import csv
import io
import json
import re
import sys
from typing import Any

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
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
# Warnings for commands that write their payload to stdout (e.g. `anki` piping
# into a file or another tool) go here so stdout stays machine-readable.
err_console = Console(stderr=True)

# Shared styling scales for the importance/difficulty badges used by the
# concepts and prerequisites renderers. Kept here so both renderers - and any
# future one - stay visually consistent instead of redefining their own.
_IMPORTANCE_ORDER = {"high": 0, "medium": 1, "low": 2}
_IMPORTANCE_BORDER = {"high": "red", "medium": "yellow", "low": "green"}
_IMPORTANCE_BADGE = {
    "high": "[bold red]● High[/bold red]",
    "medium": "[bold yellow]◑ Medium[/bold yellow]",
    "low": "[dim green]○ Low[/dim green]",
}
_DIFFICULTY_BADGE = {
    "beginner": "[bold green]▲ Beginner[/bold green]",
    "low": "[bold green]▲ Low[/bold green]",
    "medium": "[bold yellow]▲▲ Medium[/bold yellow]",
    "high": "[bold red]▲▲▲ High[/bold red]",
    "advanced": "[bold red]▲▲▲ Advanced[/bold red]",
}


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
    """Render structured concept index as numbered panels with importance badges."""
    if not result.concepts:
        console.print("[yellow]No concepts identified in the document.[/yellow]")
        return

    total = len(result.concepts)
    console.print(
        Panel(
            f"[bold white]{result.document}[/bold white]\n"
            f"[dim]{total} concept{'s' if total != 1 else ''} identified[/dim]",
            title="[bold cyan]Concept Index[/bold cyan]",
            border_style="cyan",
            expand=False,
            padding=(0, 2),
        )
    )
    console.print()

    # Sort: high importance first, then medium, then low
    ordered = sorted(result.concepts, key=lambda c: _IMPORTANCE_ORDER.get(c.importance.lower(), 1))

    for idx, c in enumerate(ordered, start=1):
        imp_key   = c.importance.lower()
        border    = _IMPORTANCE_BORDER.get(imp_key, "cyan")
        imp_badge = _IMPORTANCE_BADGE.get(imp_key, f"[cyan]{c.importance.capitalize()}[/cyan]")
        sections_str = "  ".join(f"[bold cyan]§{s}[/bold cyan]" for s in c.sections) if c.sections else "[dim]General[/dim]"

        lines: list[str] = []
        lines.append(f"[dim]Importance:[/dim]  {imp_badge}    [dim]Sections:[/dim]  {sections_str}")

        if c.description:
            lines.append(f"\n{c.description}")

        if c.related_concepts:
            related_str = "  ".join(f"[italic]{r}[/italic]" for r in c.related_concepts)
            lines.append(f"[dim]Related:[/dim]     {related_str}")

        if c.prerequisites:
            prereqs_str = "  ".join(f"[italic]{p}[/italic]" for p in c.prerequisites)
            lines.append(f"[dim]Requires:[/dim]    {prereqs_str}")

        console.print(
            Panel(
                "\n".join(lines),
                title=f"[bold]{idx}. {c.name}[/bold]",
                border_style=border,
                expand=False,
                padding=(0, 2),
            )
        )


# ---------------------------------------------------------------------------
# Prerequisites Renderer
# ---------------------------------------------------------------------------


def render_prerequisites_text(result: PrerequisiteResult) -> None:
    """Render prerequisite analysis as numbered panels with badges."""
    if not result.prerequisites:
        console.print("[yellow]No prerequisites identified in the document.[/yellow]")
        return

    total = len(result.prerequisites)
    console.print(
        Panel(
            f"[bold white]{result.document}[/bold white]\n"
            f"[dim]{total} prerequisite{'s' if total != 1 else ''} identified[/dim]",
            title="[bold cyan]Prerequisites[/bold cyan]",
            border_style="cyan",
            expand=False,
            padding=(0, 2),
        )
    )
    console.print()

    # Sort: high importance first, then medium, then low
    ordered = sorted(result.prerequisites, key=lambda p: _IMPORTANCE_ORDER.get(p.importance.lower(), 1))

    for idx, p in enumerate(ordered, start=1):
        imp_key   = p.importance.lower()
        diff_key  = p.difficulty.lower()
        border    = _IMPORTANCE_BORDER.get(imp_key, "cyan")
        imp_badge = _IMPORTANCE_BADGE.get(imp_key, f"[cyan]{p.importance.capitalize()}[/cyan]")
        diff_badge = _DIFFICULTY_BADGE.get(diff_key, f"[white]{p.difficulty.capitalize()}[/white]")

        lines: list[str] = []
        lines.append(f"[dim]Importance:[/dim]  {imp_badge}    [dim]Difficulty:[/dim]  {diff_badge}")

        if p.needed_for:
            lines.append(f"\n[dim]Needed for:[/dim]  {p.needed_for}")

        if p.relevant_sections:
            sections_str = "  ".join(f"[bold cyan]§{s}[/bold cyan]" for s in p.relevant_sections)
            lines.append(f"[dim]Sections:[/dim]    {sections_str}")

        if p.dependencies:
            deps_str = "  ".join(f"[italic]{d}[/italic]" for d in p.dependencies)
            lines.append(f"[dim]Depends on:[/dim]  {deps_str}")

        console.print(
            Panel(
                "\n".join(lines),
                title=f"[bold]{idx}. {p.name}[/bold]",
                border_style=border,
                expand=False,
                padding=(0, 2),
            )
        )


# ---------------------------------------------------------------------------
# Equations Renderer
# ---------------------------------------------------------------------------


def render_equations_text(result: EquationResult) -> None:
    """Render mathematical equations with variables and explanations.

    Mirrors the concepts/prerequisites renderers: one header panel for the
    document, then one bordered panel per equation (with the equation id in its
    title) followed by its variables, concepts, and explanation.
    """
    if not result.equations:
        console.print("[yellow]No mathematical equations identified in the document.[/yellow]")
        return

    total = len(result.equations)
    console.print(
        Panel(
            f"[bold white]{result.document}[/bold white]\n"
            f"[dim]{total} equation{'s' if total != 1 else ''} identified[/dim]",
            title="[bold cyan]Equation Index[/bold cyan]",
            border_style="cyan",
            expand=False,
            padding=(0, 2),
        )
    )
    console.print()

    for eq in result.equations:
        # Equation box - the id lives in the panel title to match the numbered
        # panels used by the concepts and prerequisites renderers.
        display_math = eq.readable or prettify_math(eq.latex) or eq.latex
        console.print(
            Panel(
                Text(display_math, justify="center", style="bold white"),
                title=f"[bold magenta]Equation {eq.id}[/bold magenta]",
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

        console.print()


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
    """Render comparison between two documents with structured panels."""
    # Header
    console.print(
        Panel(
            f"[bold cyan]{result.document_a}[/bold cyan]  [dim]vs[/dim]  [bold magenta]{result.document_b}[/bold magenta]",
            title="[bold white]Document Comparison[/bold white]",
            border_style="white",
            expand=False,
            padding=(0, 2),
        )
    )
    console.print()

    # Shared / Unique side-by-side table
    if result.shared_concepts or result.unique_to_a or result.unique_to_b:
        tbl = Table(show_header=True, header_style="bold", expand=False, box=None, padding=(0, 2))
        tbl.add_column("[bold green]Shared Concepts[/bold green]", style="green")
        tbl.add_column(f"[bold cyan]Only in {result.document_a}[/bold cyan]", style="cyan")
        tbl.add_column(f"[bold magenta]Only in {result.document_b}[/bold magenta]", style="magenta")
        rows = max(len(result.shared_concepts), len(result.unique_to_a), len(result.unique_to_b))
        for i in range(rows):
            tbl.add_row(
                result.shared_concepts[i] if i < len(result.shared_concepts) else "",
                result.unique_to_a[i]      if i < len(result.unique_to_a)      else "",
                result.unique_to_b[i]      if i < len(result.unique_to_b)      else "",
            )
        console.print(Panel(tbl, title="[bold]Concepts[/bold]", border_style="dim", expand=False, padding=(0, 1)))
        console.print()

    # Methodology panel
    if result.methodology_a or result.methodology_b:
        meth_lines = []
        if result.methodology_a:
            meth_lines.append(f"[bold cyan]{result.document_a}:[/bold cyan]  {result.methodology_a}")
        if result.methodology_b:
            meth_lines.append(f"[bold magenta]{result.document_b}:[/bold magenta]  {result.methodology_b}")
        console.print(
            Panel("\n".join(meth_lines), title="[bold yellow]Methodology[/bold yellow]", border_style="yellow", expand=False, padding=(0, 2))
        )
        console.print()

    # Prerequisites comparison panel
    if result.prerequisites_comparison:
        console.print(
            Panel(result.prerequisites_comparison, title="[bold yellow]Prerequisites[/bold yellow]", border_style="yellow", expand=False, padding=(0, 2))
        )
        console.print()

    # Equations comparison panel
    if result.equations_comparison:
        console.print(
            Panel(result.equations_comparison, title="[bold magenta]Mathematical Formulations[/bold magenta]", border_style="magenta", expand=False, padding=(0, 2))
        )
        console.print()

    # Key differences panel
    if result.key_differences:
        diff_lines = "\n".join(f"[red]•[/red] {d}" for d in result.key_differences)
        console.print(
            Panel(diff_lines, title="[bold red]Key Differences[/bold red]", border_style="red", expand=False, padding=(0, 2))
        )
        console.print()

    # Conclusion panel
    if result.conclusion:
        console.print(
            Panel(
                Markdown(result.conclusion),
                title="[bold green]Comparative Conclusion[/bold green]",
                border_style="green",
                expand=False,
            )
        )


# ---------------------------------------------------------------------------
# Study Mode Renderer
# ---------------------------------------------------------------------------


def render_study_mode_text(result: StudyModeResult) -> None:
    """Render study mode questions, concepts, and flashcards with structured panels."""
    # Header
    counts = []
    if result.key_concepts:
        counts.append(f"{len(result.key_concepts)} concepts")
    if result.flashcards:
        counts.append(f"{len(result.flashcards)} flashcards")
    if result.questions:
        counts.append(f"{len(result.questions)} questions")
    summary = "  •  ".join(counts) if counts else "study package"
    console.print(
        Panel(
            f"[bold white]{result.document}[/bold white]\n[dim]{summary}[/dim]",
            title="[bold cyan]Study Mode[/bold cyan]",
            border_style="cyan",
            expand=False,
            padding=(0, 2),
        )
    )
    console.print()

    # Key concepts panel
    if result.key_concepts:
        lines = "\n".join(f"[bold cyan]{i}.[/bold cyan] {kc}" for i, kc in enumerate(result.key_concepts, 1))
        console.print(
            Panel(lines, title="[bold green]Key Concepts[/bold green]", border_style="green", expand=False, padding=(0, 2))
        )
        console.print()

    # Prerequisites panel
    if result.prerequisites:
        lines = "\n".join(f"[yellow]•[/yellow] {pr}" for pr in result.prerequisites)
        console.print(
            Panel(lines, title="[bold yellow]Prerequisites to Review[/bold yellow]", border_style="yellow", expand=False, padding=(0, 2))
        )
        console.print()

    # Important equations panel
    if result.important_equations:
        lines = "\n".join(f"[magenta]•[/magenta] {eq}" for eq in result.important_equations)
        console.print(
            Panel(lines, title="[bold magenta]Important Equations & Formulas[/bold magenta]", border_style="magenta", expand=False, padding=(0, 2))
        )
        console.print()

    # Flashcards — already use panels; add count to title
    if result.flashcards:
        console.print("[bold cyan]Flashcards[/bold cyan]")
        console.print("[dim]" + "─" * 20 + "[/dim]")
        for idx, fc in enumerate(result.flashcards, 1):
            console.print(
                Panel(
                    f"[bold yellow]Q:[/bold yellow] {fc.question}\n\n[bold green]A:[/bold green] {fc.answer}",
                    title=f"[bold]Card {idx} / {len(result.flashcards)}[/bold]",
                    border_style="cyan",
                    expand=False,
                )
            )
        console.print()

    # Questions — each in its own panel
    if result.questions:
        console.print("[bold cyan]Questions & Self-Assessment[/bold cyan]")
        console.print("[dim]" + "─" * 20 + "[/dim]")
        for idx, q in enumerate(result.questions, 1):
            type_tag = f"[dim] ({q.type})[/dim]" if q.type else ""
            lines = [f"[bold]{q.question}[/bold]{type_tag}"]
            if q.options:
                lines += [f"  {opt}" for opt in q.options]
            if q.answer:
                lines.append(f"[dim green]Answer:[/dim green] {q.answer}")
            if q.explanation:
                lines.append(f"[dim]Explanation:[/dim] {q.explanation}")
            console.print(
                Panel(
                    "\n".join(lines),
                    title=f"[bold]Q{idx}[/bold]",
                    border_style="blue",
                    expand=False,
                    padding=(0, 2),
                )
            )


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
            if ev.verified is True:
                badge = " [green]✓ verified[/green]"
            elif ev.verified is False:
                badge = " [yellow]⚠ not verified[/yellow]"
            else:
                badge = ""
            score_str = f" [dim](relevance {ev.score:.2f})[/dim]" if ev.score is not None else ""
            console.print(f"• [bold]{loc_str}[/bold]{badge}{score_str}")
            if ev.excerpt:
                console.print(f"  [dim]\"{ev.excerpt}\"[/dim]")


# ---------------------------------------------------------------------------
# Anki Deck Export
# ---------------------------------------------------------------------------


def build_anki_rows(result: StudyModeResult, *, include_questions: bool = True) -> list[tuple[str, str]]:
    """Turn a study package into (front, back) rows for Anki import.

    Shared by `anki` (file export) and `drill` (interactive session) so both
    drill the same material in the same order.
    """
    rows = [(fc.question.strip(), fc.answer.strip()) for fc in result.flashcards if fc.question.strip()]
    if include_questions:
        for q in result.questions:
            if not q.question.strip():
                continue
            front = q.question.strip()
            if q.options:
                front += "\n" + "\n".join(q.options)
            back = q.answer.strip()
            if q.explanation.strip():
                back = f"{back}\n{q.explanation.strip()}".strip()
            rows.append((front, back))
    return rows


def format_anki_deck(rows: list[tuple[str, str]], fmt: str) -> str:
    """Serialize rows as an Anki-importable deck.

    TSV is Anki's default separator for plain-text imports; newlines inside a
    field become `<br>` (Anki renders HTML) because a literal newline would
    start a new note. CSV keeps real newlines inside standard quotes.
    """
    if fmt == "csv":
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerows(rows)
        return buffer.getvalue()

    def cell(text: str) -> str:
        flattened = "<br>".join(line.strip() for line in text.replace("\t", " ").splitlines())
        return flattened.strip()

    return "".join(f"{cell(front)}\t{cell(back)}\n" for front, back in rows)
