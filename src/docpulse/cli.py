"""Command-Line Interface for DocPulse."""

from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from docpulse import __version__
from docpulse.config import Config, load_config, save_config
from docpulse.llm import ICAGatewayClient
from docpulse.parsers import parse_document
from docpulse.prompts import (
    ANALYZE_PROMPT,
    EXPORT_PROMPT,
    SECTION_PROMPT,
    SOURCES_PROMPT,
    SYSTEM_PROMPT,
)

app = typer.Typer(
    name="docpulse",
    help="DocPulse: Modern CLI tool for intelligent document understanding, section analysis, and study guide generation.",
    no_args_is_help=True,
)
console = Console()


def version_callback(value: bool):
    if value:
        console.print(f"[bold cyan]DocPulse[/bold cyan] version [green]{__version__}[/green]")
        raise typer.Exit()


@app.callback()
def main(
    version: bool | None = typer.Option(
        None,
        "--version",
        "-v",
        help="Show DocPulse version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
):
    """DocPulse CLI root callback."""


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

    # Attempt to query available models dynamically
    discovered_models = []
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
    except Exception:  # noqa: BLE001 - best-effort probe; any failure just skips model discovery
        discovered_models = []

    default_model = current_config.model_id
    if discovered_models:
        console.print(f"[cyan]Available models/assistants under '{namespace}':[/cyan]")
        for m in discovered_models[:15]:
            m_id = m.get("id") or m.get("name") if isinstance(m, dict) else str(m)
            console.print(f"  • [bold]{m_id}[/bold]")
        if not default_model or default_model not in [m.get("id") for m in discovered_models if isinstance(m, dict)]:
            first_id = discovered_models[0].get("id") if isinstance(discovered_models[0], dict) else None
            if first_id:
                default_model = first_id

    model_id = Prompt.ask(
        "Enter Model / Worker ID",
        default=default_model or "ibm/granite-3-8b-instruct",
    )

    new_config = Config(
        endpoint_url=endpoint_url.strip(),
        api_key=api_key.strip(),
        namespace=namespace.strip(),
        model_id=model_id.strip(),
        temperature=current_config.temperature,
        max_tokens=current_config.max_tokens,
    )

    cfg_path = save_config(new_config)
    console.print(f"[green]✓[/green] Configuration saved to [bold]{cfg_path}[/bold]")

    # Health check verification against OpenAPI endpoint
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
    doc: Path = typer.Argument(..., help="Path to document (PDF, Markdown, or text file)", exists=True),
    raw: bool = typer.Option(False, "--raw", help="Output raw response instead of formatted Markdown panel"),
):
    """Analyze a document for prerequisites, core topics, and executive summary."""
    config = load_config()
    with console.status(f"[bold cyan]Parsing {doc.name}...", spinner="bouncingBar"):
        parsed = parse_document(doc)

    console.print(f"[dim]Document parsed: [bold]{parsed.file_name}[/bold] ({parsed.doc_type.upper()}) - {len(parsed.sections)} sections indexed.[/dim]")

    # Show TOC outline
    table = Table(title=f"Indexed Sections: {parsed.file_name}", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim", width=6)
    table.add_column("Section Title", style="cyan")
    table.add_column("Level / Page", justify="right")

    for sec in parsed.sections:
        level_page = f"Page {sec.page_number}" if sec.page_number else f"H{sec.level}"
        table.add_row(str(sec.id), sec.title, level_page)

    console.print(table)
    console.print()

    # Truncate content preview if document is extremely large for context limits
    content_to_analyze = parsed.raw_text[:16000]

    with console.status("[bold green]Generating document breakdown with LLM...", spinner="dots"):
        client = ICAGatewayClient(config)
        try:
            resp = client.chat_complete(
                system_prompt=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": ANALYZE_PROMPT.format(content=content_to_analyze)}],
            )
            result = resp.content
        except Exception as e:  # noqa: BLE001 - CLI boundary: turn any failure into a friendly message
            console.print(f"[bold red]Error communicating with LLM Gateway:[/bold red] {e}")
            raise typer.Exit(1) from e

    if raw:
        console.print(result)
    else:
        console.print(
            Panel(
                Markdown(result),
                title=f"Analysis: {parsed.file_name}",
                border_style="green",
                expand=False,
            )
        )


@app.command()
def section(
    doc: Path = typer.Argument(..., help="Path to document file", exists=True),
    sec_id: int = typer.Option(..., "--id", "-i", help="ID of the section to analyze"),
):
    """Perform a focused, in-depth analysis on a specific section by ID."""
    config = load_config()
    with console.status(f"[bold cyan]Indexing {doc.name}...", spinner="bouncingBar"):
        parsed = parse_document(doc)

    sec = parsed.get_section_by_id(sec_id)
    if not sec:
        console.print(f"[bold red]Error:[/bold red] Section ID [yellow]{sec_id}[/yellow] not found.")
        console.print(f"Available section IDs: {[s.id for s in parsed.sections]}")
        raise typer.Exit(1)

    console.print(f"[cyan]Selected Section [{sec.id}]:[/cyan] [bold]{sec.title}[/bold]")

    with console.status(f"[bold green]Analyzing section {sec_id}...", spinner="dots"):
        client = ICAGatewayClient(config)
        try:
            resp = client.chat_complete(
                system_prompt=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": SECTION_PROMPT.format(title=sec.title, content=sec.content or sec.title),
                    }
                ],
            )
            result = resp.content
        except Exception as e:  # noqa: BLE001 - CLI boundary: turn any failure into a friendly message
            console.print(f"[bold red]Error communicating with LLM Gateway:[/bold red] {e}")
            raise typer.Exit(1) from e

    console.print(
        Panel(
            Markdown(result),
            title=f"Section {sec.id}: {sec.title}",
            border_style="cyan",
            expand=False,
        )
    )


@app.command()
def sources(
    doc: Path = typer.Argument(..., help="Path to document file", exists=True),
):
    """Extract external references, cited papers, and code repositories."""
    config = load_config()
    with console.status(f"[bold cyan]Parsing {doc.name}...", spinner="bouncingBar"):
        parsed = parse_document(doc)

    content = parsed.raw_text[:16000]

    with console.status("[bold green]Extracting citations, repositories & links...", spinner="dots"):
        client = ICAGatewayClient(config)
        try:
            resp = client.chat_complete(
                system_prompt=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": SOURCES_PROMPT.format(content=content)}],
            )
            result = resp.content
        except Exception as e:  # noqa: BLE001 - CLI boundary: turn any failure into a friendly message
            console.print(f"[bold red]Error communicating with LLM Gateway:[/bold red] {e}")
            raise typer.Exit(1) from e

    console.print(
        Panel(
            Markdown(result),
            title=f"Sources & Citations: {parsed.file_name}",
            border_style="yellow",
            expand=False,
        )
    )


@app.command()
def export(
    doc: Path = typer.Argument(..., help="Path to document file", exists=True),
    format: str = typer.Option("md", "--format", "-f", help="Output format (currently 'md')"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Target output file path"),
):
    """Generate a comprehensive study guide / README from the document."""
    config = load_config()
    with console.status(f"[bold cyan]Parsing {doc.name}...", spinner="bouncingBar"):
        parsed = parse_document(doc)

    content = parsed.raw_text[:16000]

    with console.status("[bold green]Synthesizing study guide...", spinner="dots"):
        client = ICAGatewayClient(config)
        try:
            resp = client.chat_complete(
                system_prompt=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": EXPORT_PROMPT.format(doc_name=parsed.file_name, content=content)}
                ],
            )
            result = resp.content
        except Exception as e:  # noqa: BLE001 - CLI boundary: turn any failure into a friendly message
            console.print(f"[bold red]Error communicating with LLM Gateway:[/bold red] {e}")
            raise typer.Exit(1) from e

    target_path = output or Path(f"{doc.stem}_study_guide.md")
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(result)

    console.print(f"[bold green]✓ Study guide exported successfully to:[/bold green] [cyan]{target_path}[/cyan]")


if __name__ == "__main__":
    app()
