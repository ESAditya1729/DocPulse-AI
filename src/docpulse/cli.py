"""Command-Line Interface for DocPulse."""

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import typer
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from docpulse import __version__
from docpulse.analyzers import (
    build_document_map_result,
    parse_answer_with_citations,
    parse_comparison_response,
    parse_concepts_response,
    parse_equations_response,
    parse_prerequisites_response,
    parse_study_response,
)
from docpulse.cache import LLMOutcome, get_cached_or_generate
from docpulse.config import Config, load_config, save_config
from docpulse.errors import EXIT_INTERNAL, DocPulseError, DocumentError
from docpulse.llm import ICAGatewayClient
from docpulse.longdoc import prepare_content
from docpulse.models import ParsedDocument
from docpulse.parsers import parse_document
from docpulse.prompts import (
    ANALYZE_PROMPT,
    ASK_CITATIONS_PROMPT,
    ASK_FIRST_PROMPT,
    COMPARE_PROMPT,
    CONCEPTS_PROMPT,
    EQUATIONS_PROMPT,
    EXPORT_PROMPT,
    JSON_SYSTEM_PROMPT,
    MAP_PROMPT,
    PREREQUISITES_PROMPT,
    SECTION_PROMPT,
    SOURCES_PROMPT,
    STUDY_PROMPT,
    SYSTEM_PROMPT,
)
from docpulse.renderers import (
    console,
    generate_document_map_mermaid,
    print_json,
    render_answer_with_citations_text,
    render_comparison_text,
    render_concepts_text,
    render_document_map_text,
    render_equations_text,
    render_json,
    render_markdown_with_equations,
    render_prerequisites_text,
    render_study_mode_text,
)

app = typer.Typer(
    name="docpulse",
    help="DocPulse: Modern CLI tool for intelligent document understanding, section analysis, and study guide generation.",
    no_args_is_help=True,
)


@dataclass
class AppContext:
    """Global options set by the root callback and read by every command."""

    debug: bool = False
    force_text: bool = False


def version_callback(value: bool):
    if value:
        console.print(f"[bold cyan]DocPulse[/bold cyan] version [green]{__version__}[/green]")
        raise typer.Exit()


def options(ctx: typer.Context) -> AppContext:
    """Read the global options stashed by the root callback."""
    return ctx.obj if isinstance(ctx.obj, AppContext) else AppContext()


def make_client(config: Config, opts: AppContext, is_json: bool) -> ICAGatewayClient:
    """Build a gateway client, announcing retries unless output must stay clean."""
    if not config.api_key:
        console.print(
            "[yellow]Note:[/yellow] No API key configured. "
            "This is fine for keyless gateways; otherwise requests may fail with 401/403."
        )

    if is_json or opts.debug:
        return ICAGatewayClient(config)

    def on_retry(attempt: int, attempts: int, delay: float, error: DocPulseError) -> None:
        reason = error.message.splitlines()[0]
        console.print(f"[dim]Retrying ({attempt}/{attempts}) in {delay:.1f}s - {reason}[/dim]")

    return ICAGatewayClient(config, on_retry=on_retry)


def fail(
    exc: BaseException,
    *,
    is_json: bool,
    opts: AppContext,
    context: dict[str, Any] | None = None,
) -> NoReturn:
    """Report a failure in the command's output mode and exit with its code."""
    payload: dict[str, Any] = dict(context or {})

    if isinstance(exc, DocPulseError):
        if is_json:
            print_json({"error": exc.message, "hint": exc.hint, **payload})
        else:
            console.print(f"[bold red]Error:[/bold red] {exc.message}")
            if exc.hint:
                console.print(f"[dim]{exc.hint}[/dim]")
        raise typer.Exit(exc.exit_code) from exc

    if opts.debug:
        console.print_exception()
    message = f"{type(exc).__name__}: {exc}"
    if is_json:
        print_json({"error": message, "unexpected": True, **payload})
    else:
        console.print(f"[bold red]Unexpected error:[/bold red] {message}")
        if not opts.debug:
            console.print("[dim]Re-run with --debug for the full traceback.[/dim]")
    raise typer.Exit(EXIT_INTERNAL) from exc


def show_warnings(warnings: list[str], *, is_json: bool) -> None:
    """Print collected warnings; JSON commands carry them in the payload instead."""
    if is_json:
        return
    for warning in warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")


def truncation_warning(config: Config) -> str:
    return (
        f"Response was cut off by the model's token limit (max_tokens={config.max_tokens}). "
        "The output above is incomplete - raise max_tokens in your DocPulse config and re-run with --no-cache."
    )


def chunk_mapper(client: ICAGatewayClient, config: Config, no_cache: bool) -> Callable[[str, int, int], str]:
    """Build the per-chunk condensation callback used for large documents."""

    def map_chunk(chunk: str, index: int, total: int) -> str:
        outcome = get_cached_or_generate(
            client,
            config,
            system_prompt=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": MAP_PROMPT.format(index=index, total=total, content=chunk)}],
            use_cache=not no_cache,
        )
        return outcome.content

    return map_chunk


def load_document(doc: Path, opts: AppContext, is_json: bool) -> ParsedDocument:
    """Parse a document, showing a spinner only in human-readable mode."""
    if not is_json:
        with console.status(f"[bold cyan]Parsing {doc.name}...", spinner="bouncingBar"):
            return parse_document(doc, force_text=opts.force_text)
    return parse_document(doc, force_text=opts.force_text)


def generate_from_document(
    client: ICAGatewayClient,
    config: Config,
    source_text: str,
    build_prompt: Callable[[str], str],
    *,
    status_label: str,
    no_cache: bool,
    system_prompt: str = SYSTEM_PROMPT,
    quiet: bool = False,
) -> tuple[str, list[str]]:
    """Prepare source_text (chunking it if large) and run the final LLM call.

    Returns (result, warnings) so callers can tell the user that the response
    was cut short or that part of a very large document went unread.
    """

    def map_chunk(chunk: str, index: int, total: int) -> str:
        outcome = get_cached_or_generate(
            client,
            config,
            system_prompt=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": MAP_PROMPT.format(index=index, total=total, content=chunk)}],
            use_cache=not no_cache,
        )
        return outcome.content

    prepared = prepare_content(
        map_chunk,
        source_text,
        on_progress=(lambda i, n: console.print(f"[dim]Summarizing excerpt {i}/{n}...[/dim]")) if not quiet else None,
    )
    warnings: list[str] = []
    if prepared.chunked and not quiet:
        console.print(f"[dim]Large document - condensed from {prepared.chunk_count} chunks for full coverage.[/dim]")
    if prepared.truncated:
        warnings.append(
            f"Document is larger than the {prepared.chunk_count}-chunk ceiling; trailing content was not analyzed."
        )
        if not quiet:
            console.print(f"[yellow]Warning:[/yellow] {warnings[-1]}")

    if quiet:
        outcome = get_cached_or_generate(
            client,
            config,
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": build_prompt(prepared.text)}],
            use_cache=not no_cache,
        )
    else:
        with console.status(status_label, spinner="dots"):
            outcome = get_cached_or_generate(
                client,
                config,
                system_prompt=system_prompt,
                messages=[{"role": "user", "content": build_prompt(prepared.text)}],
                use_cache=not no_cache,
            )

    if outcome.truncated:
        warnings.append(truncation_warning(config))
    if outcome.was_cached and not quiet:
        console.print("[dim]⚡ Served from cache (use --no-cache to force a fresh run).[/dim]")

    return outcome.content, warnings


@app.callback()
def main(
    ctx: typer.Context,
    version: bool | None = typer.Option(
        None,
        "--version",
        "-v",
        help="Show DocPulse version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
    debug: bool = typer.Option(False, "--debug", help="Print full tracebacks for unexpected errors."),
    force_text: bool = typer.Option(
        False,
        "--force-text",
        help="Read the document as plain text, bypassing file-type detection.",
    ),
):
    """DocPulse CLI root callback."""
    ctx.obj = AppContext(debug=debug, force_text=force_text)


@app.command()
def init():
    """Interactive connection setup & health-check verification for IBM ICA Gateway."""
    console.print(
        Panel.fit(
            "[bold cyan]DocPulse Initialization & ICA Gateway Setup[/bold cyan]\n"
            "Configure your endpoint URL and credentials to connect to IBM ICA Gateway.",
            border_style="cyan",
        )
    )

    current_config = load_config()

    endpoint_url = Prompt.ask(
        "Enter IBM ICA Gateway Endpoint URL",
        default=current_config.endpoint_url,
    )
    api_key = Prompt.ask(
        "Enter IBM ICA Developer API Key",
        default=current_config.api_key or "",
        password=True,
    )
    namespace = Prompt.ask(
        "Enter ICA Namespace (chat-models, assistants, agents, digital-workforce)",
        default=current_config.namespace,
        choices=["chat-models", "assistants", "agents", "digital-workforce"],
    )

    temp_client = ICAGatewayClient(
        Config(
            endpoint_url=endpoint_url.strip(),
            api_key=api_key.strip(),
            namespace=namespace.strip(),
            model_id="",
        )
    )
    try:
        discovered_models = temp_client.list_models()
    except Exception:  # noqa: BLE001 - discovery is a convenience; the health check below is authoritative
        discovered_models = []

    valid_model_ids = {m.get("id") or m.get("name") for m in discovered_models if isinstance(m, dict)} - {None}

    default_model = current_config.model_id
    if discovered_models:
        console.print(f"[cyan]Available models/assistants under '{namespace}':[/cyan]")
        for m in discovered_models[:15]:
            m_id = m.get("id") or m.get("name") if isinstance(m, dict) else str(m)
            console.print(f"  • [bold]{m_id}[/bold]")
        if not default_model or default_model not in valid_model_ids:
            first_id = discovered_models[0].get("id") if isinstance(discovered_models[0], dict) else None
            if first_id:
                default_model = first_id

    while True:
        model_id = Prompt.ask(
            "Enter Model / Worker ID",
            default=default_model or "ibm/granite-3-8b-instruct",
        ).strip()

        if valid_model_ids and model_id not in valid_model_ids:
            console.print(
                f"[yellow]Warning:[/yellow] '{model_id}' wasn't in the models discovered for namespace "
                f"'{namespace}' - it may belong to a different namespace or be mistyped."
            )
            if Confirm.ask("Use it anyway?", default=False):
                break
        else:
            break

    new_config = current_config.model_copy(
        update={
            "endpoint_url": endpoint_url.strip(),
            "api_key": api_key.strip(),
            "namespace": namespace.strip(),
            "model_id": model_id.strip(),
        }
    )

    cfg_path = save_config(new_config)
    console.print(f"[green]✓[/green] Configuration saved to [bold]{cfg_path}[/bold]")

    with console.status("[bold green]Verifying connection & gateway health...", spinner="dots"):
        client = ICAGatewayClient(new_config)
        health = client.check_health()

    if health["status"] == "ok":
        models_info = f" ({health.get('models_count', 0)} items available)" if "models_count" in health else ""
        console.print(f"[bold green]✓ Connection successful![/bold green] Endpoint: {health.get('endpoint')}{models_info}")
    elif health["status"] == "warning":
        console.print(f"[bold yellow]! Notice:[/bold yellow] {health.get('message')}")
    else:
        console.print(f"[bold red]✗ Connection check failed:[/bold red] {health.get('message')}")
        console.print("[dim]You can still use local document parsing or re-run 'docpulse init' when ready.[/dim]")


@app.command()
def analyze(
    ctx: typer.Context,
    doc: Path = typer.Argument(..., help="Path to document (PDF, Markdown, or text file)"),
    raw: bool = typer.Option(False, "--raw", help="Output raw response instead of formatted Markdown panel"),
    format: str = typer.Option("text", "--format", "-f", help="Output format: 'text' or 'json'"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the response cache and force a fresh LLM call"),
):
    """Analyze a document for prerequisites, core topics, and executive summary."""
    opts = options(ctx)
    config = load_config()
    is_json = format.lower() == "json"
    context = {"document": doc.name, "command": "analyze"}

    try:
        config.validate_for_command()
        client = make_client(config, opts, is_json)
        parsed = load_document(doc, opts, is_json)

        if not is_json:
            console.print(
                f"[dim]Document parsed: [bold]{parsed.file_name}[/bold] ({parsed.doc_type.upper()})"
                f" - {len(parsed.sections)} sections indexed.[/dim]"
            )

            table = Table(title=f"Indexed Sections: {parsed.file_name}", show_header=True, header_style="bold magenta")
            table.add_column("ID", style="dim", width=6)
            table.add_column("Section Title", style="cyan")
            table.add_column("Level / Page", justify="right")

            for sec in parsed.sections:
                level_page = f"Page {sec.page_number}" if sec.page_number else f"H{sec.level}"
                table.add_row(str(sec.id), sec.title, level_page)

            console.print(table)
            console.print()

        result, warnings = generate_from_document(
            client,
            config,
            parsed.raw_text,
            lambda content: ANALYZE_PROMPT.format(content=content),
            status_label="[bold green]Generating document breakdown with LLM...",
            no_cache=no_cache,
            quiet=is_json,
        )
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001 - every failure path funnels through fail()
        fail(exc, is_json=is_json, opts=opts, context=context)

    if is_json:
        print_json(
            {
                "document": parsed.file_name,
                "sections_count": len(parsed.sections),
                "analysis": result,
                "warnings": warnings,
            }
        )
    else:
        show_warnings(warnings, is_json=is_json)
        if raw:
            console.print(result)
        else:
            console.print(
                Panel(
                    render_markdown_with_equations(result),
                    title=f"Analysis: {parsed.file_name}",
                    border_style="green",
                    expand=False,
                )
            )


@app.command()
def section(
    ctx: typer.Context,
    doc: Path = typer.Argument(..., help="Path to document file"),
    sec_id: int = typer.Option(..., "--id", "-i", help="ID of the section to analyze"),
    format: str = typer.Option("text", "--format", "-f", help="Output format: 'text' or 'json'"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the response cache and force a fresh LLM call"),
):
    """Perform a focused, in-depth analysis on a specific section by ID."""
    opts = options(ctx)
    config = load_config()
    is_json = format.lower() == "json"
    context = {"document": doc.name, "command": "section", "section_id": sec_id}

    try:
        config.validate_for_command()
        client = make_client(config, opts, is_json)
        parsed = load_document(doc, opts, is_json)

        sec = parsed.get_section_by_id(sec_id)
        if not sec:
            available = [s.id for s in parsed.sections]
            if is_json:
                print_json({"error": f"Section ID {sec_id} not found", "available_ids": available, **context})
            else:
                console.print(f"[bold red]Error:[/bold red] Section ID [yellow]{sec_id}[/yellow] not found.")
                console.print(f"Available section IDs: {available}")
            raise typer.Exit(DocumentError.exit_code)

        if not is_json:
            console.print(f"[cyan]Selected Section [{sec.id}]:[/cyan] [bold]{sec.title}[/bold]")

        result, warnings = generate_from_document(
            client,
            config,
            sec.content or sec.title,
            lambda content: SECTION_PROMPT.format(title=sec.title, content=content),
            status_label=f"[bold green]Analyzing section {sec_id}...",
            no_cache=no_cache,
            quiet=is_json,
        )
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        fail(exc, is_json=is_json, opts=opts, context=context)

    if is_json:
        print_json(
            {
                "document": parsed.file_name,
                "section_id": sec.id,
                "section_title": sec.title,
                "analysis": result,
                "warnings": warnings,
            }
        )
    else:
        show_warnings(warnings, is_json=is_json)
        console.print(
            Panel(
                render_markdown_with_equations(result),
                title=f"Section {sec.id}: {sec.title}",
                border_style="cyan",
                expand=False,
            )
        )


@app.command()
def concepts(
    ctx: typer.Context,
    doc: Path = typer.Argument(..., help="Path to document file"),
    name: str | None = typer.Option(None, "--name", "-n", help="Filter for a specific concept by name"),
    format: str = typer.Option("text", "--format", "-f", help="Output format: 'text' or 'json'"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the response cache and force a fresh LLM call"),
):
    """Extract key concepts from the entire document and provide a structured concept index."""
    opts = options(ctx)
    config = load_config()
    is_json = format.lower() == "json"
    context = {"document": doc.name, "command": "concepts"}

    try:
        config.validate_for_command()
        client = make_client(config, opts, is_json)
        parsed = load_document(doc, opts, is_json)

        raw_result, warnings = generate_from_document(
            client,
            config,
            parsed.raw_text,
            lambda content: CONCEPTS_PROMPT.format(content=content),
            status_label="[bold green]Extracting document concepts...",
            no_cache=no_cache,
            system_prompt=JSON_SYSTEM_PROMPT,
            quiet=is_json,
        )
        result = parse_concepts_response(raw_result, parsed.file_name, warnings)

        # Apply name filter if requested
        if name:
            target = name.strip().lower()
            result.concepts = [c for c in result.concepts if target in c.name.lower()]
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        fail(exc, is_json=is_json, opts=opts, context=context)

    if is_json:
        print_json({**result.model_dump(), "warnings": warnings})
    else:
        show_warnings(warnings, is_json=is_json)
        render_concepts_text(result)


@app.command()
def prerequisites(
    ctx: typer.Context,
    doc: Path = typer.Argument(..., help="Path to document file"),
    format: str = typer.Option("text", "--format", "-f", help="Output format: 'text' or 'json'"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the response cache and force a fresh LLM call"),
):
    """Analyze prerequisite background knowledge and dependencies needed for the document."""
    opts = options(ctx)
    config = load_config()
    is_json = format.lower() == "json"
    context = {"document": doc.name, "command": "prerequisites"}

    try:
        config.validate_for_command()
        client = make_client(config, opts, is_json)
        parsed = load_document(doc, opts, is_json)

        raw_result, warnings = generate_from_document(
            client,
            config,
            parsed.raw_text,
            lambda content: PREREQUISITES_PROMPT.format(content=content),
            status_label="[bold green]Analyzing prerequisites and dependencies...",
            no_cache=no_cache,
            system_prompt=JSON_SYSTEM_PROMPT,
            quiet=is_json,
        )
        result = parse_prerequisites_response(raw_result, parsed.file_name, warnings)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        fail(exc, is_json=is_json, opts=opts, context=context)

    if is_json:
        print_json({**result.model_dump(), "warnings": warnings})
    else:
        show_warnings(warnings, is_json=is_json)
        render_prerequisites_text(result)


@app.command()
def equations(
    ctx: typer.Context,
    doc: Path = typer.Argument(..., help="Path to document file"),
    format: str = typer.Option("text", "--format", "-f", help="Output format: 'text' or 'json'"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the response cache and force a fresh LLM call"),
):
    """Identify, index, and explain mathematical equations in the document."""
    opts = options(ctx)
    config = load_config()
    is_json = format.lower() == "json"
    context = {"document": doc.name, "command": "equations"}

    try:
        config.validate_for_command()
        client = make_client(config, opts, is_json)
        parsed = load_document(doc, opts, is_json)

        raw_result, warnings = generate_from_document(
            client,
            config,
            parsed.raw_text,
            lambda content: EQUATIONS_PROMPT.format(content=content),
            status_label="[bold green]Extracting and explaining equations...",
            no_cache=no_cache,
            system_prompt=JSON_SYSTEM_PROMPT,
            quiet=is_json,
        )
        result = parse_equations_response(raw_result, parsed.file_name, warnings)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        fail(exc, is_json=is_json, opts=opts, context=context)

    if is_json:
        print_json({**result.model_dump(), "warnings": warnings})
    else:
        show_warnings(warnings, is_json=is_json)
        render_equations_text(result)


@app.command()
def sources(
    ctx: typer.Context,
    doc: Path = typer.Argument(..., help="Path to document file"),
    format: str = typer.Option("text", "--format", "-f", help="Output format: 'text' or 'json'"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the response cache and force a fresh LLM call"),
):
    """Extract external references, cited papers, and code repositories."""
    opts = options(ctx)
    config = load_config()
    is_json = format.lower() == "json"
    context = {"document": doc.name, "command": "sources"}

    try:
        config.validate_for_command()
        client = make_client(config, opts, is_json)
        parsed = load_document(doc, opts, is_json)

        result, warnings = generate_from_document(
            client,
            config,
            parsed.raw_text,
            lambda content: SOURCES_PROMPT.format(content=content),
            status_label="[bold green]Extracting citations, repositories & links...",
            no_cache=no_cache,
            quiet=is_json,
        )
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        fail(exc, is_json=is_json, opts=opts, context=context)

    if is_json:
        print_json({"document": parsed.file_name, "sources": result, "warnings": warnings})
    else:
        show_warnings(warnings, is_json=is_json)
        console.print(
            Panel(
                render_markdown_with_equations(result),
                title=f"Sources & Citations: {parsed.file_name}",
                border_style="yellow",
                expand=False,
            )
        )


@app.command("map")
def document_map(
    ctx: typer.Context,
    doc: Path = typer.Argument(..., help="Path to document file"),
    format: str = typer.Option("text", "--format", "-f", help="Output format: 'text', 'json', or 'mermaid'"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the response cache and force a fresh LLM call"),
):
    """Generate a unified structural map connecting sections, concepts, prerequisites, and equations."""
    opts = options(ctx)
    config = load_config()
    fmt = format.lower()
    is_json = fmt == "json"
    is_non_text = fmt in ("json", "mermaid")
    context = {"document": doc.name, "command": "map"}

    try:
        config.validate_for_command()
        client = make_client(config, opts, is_non_text)
        parsed = load_document(doc, opts, is_non_text)

        warnings: list[str] = []
        raw_c, w1 = generate_from_document(
            client,
            config,
            parsed.raw_text,
            lambda content: CONCEPTS_PROMPT.format(content=content),
            status_label="[bold green]Mapping concepts...",
            no_cache=no_cache,
            system_prompt=JSON_SYSTEM_PROMPT,
            quiet=is_non_text,
        )
        warnings += w1
        concepts_res = parse_concepts_response(raw_c, parsed.file_name, warnings)

        raw_p, w2 = generate_from_document(
            client,
            config,
            parsed.raw_text,
            lambda content: PREREQUISITES_PROMPT.format(content=content),
            status_label="[bold green]Mapping prerequisites...",
            no_cache=no_cache,
            system_prompt=JSON_SYSTEM_PROMPT,
            quiet=is_non_text,
        )
        warnings += w2
        prereqs_res = parse_prerequisites_response(raw_p, parsed.file_name, warnings)

        raw_e, w3 = generate_from_document(
            client,
            config,
            parsed.raw_text,
            lambda content: EQUATIONS_PROMPT.format(content=content),
            status_label="[bold green]Mapping equations...",
            no_cache=no_cache,
            system_prompt=JSON_SYSTEM_PROMPT,
            quiet=is_non_text,
        )
        warnings += w3
        eqs_res = parse_equations_response(raw_e, parsed.file_name, warnings)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        fail(exc, is_json=is_json, opts=opts, context=context)

    doc_map = build_document_map_result(
        parsed,
        concepts=concepts_res.concepts,
        prerequisites=prereqs_res.prerequisites,
        equations=eqs_res.equations,
        references=[],
    )

    if fmt == "json":
        print_json({**doc_map.model_dump(), "warnings": warnings})
    elif fmt == "mermaid":
        show_warnings(warnings, is_json=False)
        console.print(generate_document_map_mermaid(doc_map))
    else:
        show_warnings(warnings, is_json=False)
        render_document_map_text(doc_map)


@app.command()
def compare(
    ctx: typer.Context,
    doc1: Path = typer.Argument(..., help="Path to first document file"),
    doc2: Path = typer.Argument(..., help="Path to second document file"),
    format: str = typer.Option("text", "--format", "-f", help="Output format: 'text' or 'json'"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the response cache and force a fresh LLM call"),
):
    """Compare two documents structurally and conceptually."""
    opts = options(ctx)
    config = load_config()
    is_json = format.lower() == "json"
    context = {"document_a": doc1.name, "document_b": doc2.name, "command": "compare"}

    try:
        config.validate_for_command()
        client = make_client(config, opts, is_json)

        if not is_json:
            with console.status("[bold cyan]Parsing documents...", spinner="bouncingBar"):
                parsed1 = load_document(doc1, opts, is_json)
                parsed2 = load_document(doc2, opts, is_json)
        else:
            parsed1 = load_document(doc1, opts, is_json)
            parsed2 = load_document(doc2, opts, is_json)

        mapper = chunk_mapper(client, config, no_cache)
        prep1 = prepare_content(mapper, parsed1.raw_text)
        prep2 = prepare_content(mapper, parsed2.raw_text)

        prompt = COMPARE_PROMPT.format(
            doc_a_name=parsed1.file_name,
            doc_a_content=prep1.text,
            doc_b_name=parsed2.file_name,
            doc_b_content=prep2.text,
        )

        if is_json:
            outcome = get_cached_or_generate(
                client, config, system_prompt=JSON_SYSTEM_PROMPT, messages=[{"role": "user", "content": prompt}], use_cache=not no_cache
            )
        else:
            with console.status("[bold green]Comparing documents...", spinner="dots"):
                outcome = get_cached_or_generate(
                    client, config, system_prompt=JSON_SYSTEM_PROMPT, messages=[{"role": "user", "content": prompt}], use_cache=not no_cache
                )
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        fail(exc, is_json=is_json, opts=opts, context=context)

    warnings: list[str] = []
    result = parse_comparison_response(outcome.content, parsed1.file_name, parsed2.file_name, warnings)
    if outcome.truncated:
        warnings.append(truncation_warning(config))

    if is_json:
        print_json({**result.model_dump(), "warnings": warnings})
    else:
        show_warnings(warnings, is_json=is_json)
        render_comparison_text(result)


@app.command()
def study(
    ctx: typer.Context,
    doc: Path = typer.Argument(..., help="Path to document file"),
    questions: int = typer.Option(5, "--questions", "-q", help="Number of questions to generate"),
    format: str = typer.Option("text", "--format", "-f", help="Output format: 'text' or 'json'"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the response cache and force a fresh LLM call"),
):
    """Generate interactive study material, key topics, flashcards, and self-assessment questions."""
    opts = options(ctx)
    config = load_config()
    is_json = format.lower() == "json"
    context = {"document": doc.name, "command": "study"}

    try:
        config.validate_for_command()
        client = make_client(config, opts, is_json)
        parsed = load_document(doc, opts, is_json)

        raw_result, warnings = generate_from_document(
            client,
            config,
            parsed.raw_text,
            lambda content: STUDY_PROMPT.format(num_questions=questions, content=content),
            status_label="[bold green]Generating study package & flashcards...",
            no_cache=no_cache,
            system_prompt=JSON_SYSTEM_PROMPT,
            quiet=is_json,
        )
        result = parse_study_response(raw_result, parsed.file_name, warnings)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        fail(exc, is_json=is_json, opts=opts, context=context)

    if is_json:
        print_json({**result.model_dump(), "warnings": warnings})
    else:
        show_warnings(warnings, is_json=is_json)
        render_study_mode_text(result)


@app.command()
def export(
    ctx: typer.Context,
    doc: Path = typer.Argument(..., help="Path to document file"),
    format: str = typer.Option("md", "--format", "-f", help="Output format ('md' or 'json')"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Target output file path"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the response cache and force a fresh LLM call"),
):
    """Generate a comprehensive study guide / README from the document."""
    opts = options(ctx)
    config = load_config()
    is_json = format.lower() == "json"
    context = {"document": doc.name, "command": "export"}

    try:
        config.validate_for_command()
        client = make_client(config, opts, is_json)
        parsed = load_document(doc, opts, is_json)

        result, warnings = generate_from_document(
            client,
            config,
            parsed.raw_text,
            lambda content: EXPORT_PROMPT.format(doc_name=parsed.file_name, content=content),
            status_label="[bold green]Synthesizing study guide...",
            no_cache=no_cache,
            quiet=is_json,
        )

        if is_json:
            payload = render_json({"document": parsed.file_name, "content": result, "format": "markdown", "warnings": warnings})
        else:
            payload = result
        target_path = output or Path(f"{doc.stem}_study_guide.md")
        if output or not is_json:
            try:
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(payload)
            except OSError as exc:
                raise DocumentError(f"Could not write to {target_path}: {exc}") from exc
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        fail(exc, is_json=is_json, opts=opts, context=context)

    if is_json and not output:
        sys.stdout.write(payload + "\n")
        return

    show_warnings(warnings, is_json=False)
    console.print(f"[bold green]✓ Study guide exported successfully to:[/bold green] [cyan]{target_path}[/cyan]")


@app.command()
def ask(
    ctx: typer.Context,
    doc: Path = typer.Argument(..., help="Path to document file"),
    question: str | None = typer.Argument(
        None, help="Ask a single question and exit; omit it for an interactive session"
    ),
    sec_id: int | None = typer.Option(
        None, "--id", "-i", help="Restrict Q&A to one section instead of the whole document"
    ),
    citations: bool = typer.Option(False, "--citations", "-c", help="Extract and show grounding evidence & citations"),
    format: str = typer.Option("text", "--format", "-f", help="Output format: 'text' or 'json'"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the response cache and force a fresh LLM call"),
):
    """Ask follow-up questions about a document, one-shot or interactively, with optional citations."""
    opts = options(ctx)
    config = load_config()
    is_json = format.lower() == "json"
    context = {"document": doc.name, "command": "ask"}

    try:
        config.validate_for_command()
        client = make_client(config, opts, is_json)
        parsed = load_document(doc, opts, is_json)

        if sec_id is not None:
            sec = parsed.get_section_by_id(sec_id)
            if not sec:
                if is_json:
                    print_json({"error": f"Section ID {sec_id} not found", **context})
                else:
                    console.print(f"[bold red]Error:[/bold red] Section ID [yellow]{sec_id}[/yellow] not found.")
                    console.print(f"Available section IDs: {[s.id for s in parsed.sections]}")
                raise typer.Exit(DocumentError.exit_code)
            doc_label = f"{parsed.file_name} (section {sec.id}: {sec.title})"
            source_text = sec.content or sec.title
        else:
            doc_label = parsed.file_name
            source_text = parsed.raw_text

        mapper = chunk_mapper(client, config, no_cache)

        if not is_json:
            with console.status("[bold cyan]Preparing document context...", spinner="dots"):
                prepared = prepare_content(
                    mapper,
                    source_text,
                    on_progress=lambda i, n: console.print(f"[dim]Summarizing excerpt {i}/{n}...[/dim]"),
                )
        else:
            prepared = prepare_content(mapper, source_text)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        fail(exc, is_json=is_json, opts=opts, context=context)

    warnings: list[str] = []
    if prepared.truncated:
        warnings.append(
            f"Document is larger than the {prepared.chunk_count}-chunk ceiling; trailing content was not included."
        )

    messages: list[dict[str, str]] = []
    first_turn = True

    def ask_once(user_question: str) -> LLMOutcome:
        nonlocal first_turn
        sys_prompt = JSON_SYSTEM_PROMPT if citations else SYSTEM_PROMPT
        if first_turn:
            if citations:
                prompt_content = ASK_CITATIONS_PROMPT.format(
                    doc_name=doc_label, content=prepared.text, question=user_question
                )
            else:
                prompt_content = ASK_FIRST_PROMPT.format(
                    doc_name=doc_label, content=prepared.text, question=user_question
                )
            messages.append({"role": "user", "content": prompt_content})
            first_turn = False
        else:
            messages.append({"role": "user", "content": user_question})

        outcome = get_cached_or_generate(
            client, config, system_prompt=sys_prompt, messages=messages, use_cache=not no_cache
        )
        messages.append({"role": "assistant", "content": outcome.content})
        return outcome

    def print_result(outcome: LLMOutcome) -> None:
        turn_warnings = list(warnings)
        if outcome.truncated:
            turn_warnings.append(truncation_warning(config))

        if citations:
            answer_warnings: list[str] = []
            evidence_result = parse_answer_with_citations(outcome.content, parsed, answer_warnings)
            turn_warnings += answer_warnings
            evidence_result.was_cached = outcome.was_cached
            if is_json:
                print_json({**evidence_result.model_dump(), "warnings": turn_warnings})
            else:
                show_warnings(turn_warnings, is_json=False)
                render_answer_with_citations_text(evidence_result)
                if outcome.was_cached:
                    console.print("[dim]⚡ Served from cache (use --no-cache to force a fresh run).[/dim]")
        else:
            if is_json:
                print_json(
                    {"answer": outcome.content, "was_cached": outcome.was_cached, "warnings": turn_warnings}
                )
            else:
                show_warnings(turn_warnings, is_json=False)
                console.print(
                    Panel(render_markdown_with_equations(outcome.content), title="Answer", border_style="blue", expand=False)
                )
                if outcome.was_cached:
                    console.print("[dim]⚡ Served from cache (use --no-cache to force a fresh run).[/dim]")

    if question:
        try:
            outcome = ask_once(question)
        except typer.Exit:
            raise
        except Exception as exc:  # noqa: BLE001
            fail(exc, is_json=is_json, opts=opts, context=context)
        else:
            print_result(outcome)
        return

    console.print(f"[bold cyan]Ask questions about {doc_label}.[/bold cyan] Type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_question = Prompt.ask("[bold green]You[/bold green]")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        stripped = user_question.strip()
        if not stripped or stripped.lower() in {"exit", "quit"}:
            break

        try:
            outcome = ask_once(stripped)
        except typer.Exit:
            raise
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, DocPulseError):
                console.print(f"[bold red]Error:[/bold red] {exc.message}")
                if exc.hint:
                    console.print(f"[dim]{exc.hint}[/dim]")
            else:
                console.print(f"[bold red]Unexpected error:[/bold red] {type(exc).__name__}: {exc}")
            continue
        print_result(outcome)


if __name__ == "__main__":
    app()
