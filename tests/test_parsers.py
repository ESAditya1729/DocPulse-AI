"""Test suite for DocPulse parsers and configuration."""

from pathlib import Path

from docpulse.config import Config, load_config, save_config
from docpulse.parsers import parse_document
from docpulse.parsers.markdown_parser import MarkdownParser


def test_config_save_and_load(tmp_path: Path, monkeypatch):
    custom_cfg_path = tmp_path / "config.json"
    monkeypatch.setenv("DOCPULSE_CONFIG_PATH", str(custom_cfg_path))

    cfg = Config(
        endpoint_url="https://test.ica.ibm.com/v1",
        api_key="secret-key",
        namespace="assistants",
        model_id="ibm/granite-3-8b",
        temperature=0.5,
    )
    save_config(cfg)
    loaded = load_config()

    assert loaded.endpoint_url == "https://test.ica.ibm.com/v1"
    assert loaded.api_key == "secret-key"
    assert loaded.namespace == "assistants"
    assert loaded.model_id == "ibm/granite-3-8b"
    assert loaded.temperature == 0.5


def test_markdown_parser(tmp_path: Path):
    sample_md = tmp_path / "sample.md"
    content = """# Introduction
Welcome to DocPulse test doc.

## Architecture
DocPulse uses Typer and Rich.

## Data Processing
Section extraction and token parsing.
"""
    sample_md.write_text(content, encoding="utf-8")

    parser = MarkdownParser()
    doc = parser.parse(sample_md)

    assert doc.file_name == "sample.md"
    assert doc.doc_type == "markdown"
    assert len(doc.sections) == 3
    assert doc.sections[0].title == "Introduction"
    assert doc.sections[1].title == "Architecture"
    assert doc.sections[2].title == "Data Processing"

    sec2 = doc.get_section_by_id(2)
    assert sec2 is not None
    assert sec2.title == "Architecture"
    assert "Typer and Rich" in sec2.content


def test_parser_factory(tmp_path: Path):
    sample_md = tmp_path / "doc.md"
    sample_md.write_text("# Test", encoding="utf-8")
    
    doc = parse_document(sample_md)
    assert doc.doc_type == "markdown"
