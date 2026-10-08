"""CLI-level tests for error handling, exit codes, and honesty about failures.

These exercise the paths a user hits when something is wrong: no configuration,
an unreadable document, or a response the model cut short.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from docpulse.cli import app
from docpulse.errors import EXIT_DOCUMENT, EXIT_GATEWAY, EXIT_USAGE
from docpulse.llm import LLMResponse

runner = CliRunner()


def write_doc(tmp_path, name="doc.md", body="# Title\nSome real content about attention."):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def unconfigured(tmp_path, monkeypatch):
    """Point DocPulse at a config file with no endpoint, as a fresh install has."""
    empty = tmp_path / "unconfigured.json"
    empty.write_text(json.dumps({"api_key": ""}), encoding="utf-8")
    monkeypatch.setenv("DOCPULSE_CONFIG_PATH", str(empty))


# --- Configuration ----------------------------------------------------------


def test_unconfigured_run_exits_2_with_init_instructions(tmp_path, monkeypatch):
    unconfigured(tmp_path, monkeypatch)

    result = runner.invoke(app, ["analyze", str(write_doc(tmp_path))])

    assert result.exit_code == EXIT_USAGE
    assert "not configured" in result.output
    assert "docpulse init" in result.output


def test_unconfigured_json_run_emits_a_machine_readable_error(tmp_path, monkeypatch):
    unconfigured(tmp_path, monkeypatch)

    result = runner.invoke(app, ["analyze", str(write_doc(tmp_path)), "--format", "json"])

    assert result.exit_code == EXIT_USAGE
    payload = json.loads(result.output)
    assert "not configured" in payload["error"]
    assert "docpulse init" in payload["hint"]


# --- Document problems ------------------------------------------------------


def test_missing_document_exits_4(tmp_path):
    result = runner.invoke(app, ["analyze", str(tmp_path / "absent.md")])

    assert result.exit_code == EXIT_DOCUMENT
    assert "not found" in result.output


def test_unsupported_document_type_exits_4_and_lists_supported_types(tmp_path):
    docx = tmp_path / "report.docx"
    docx.write_bytes(b"PK\x03\x04 not really a docx")

    result = runner.invoke(app, ["analyze", str(docx)])

    assert result.exit_code == EXIT_DOCUMENT
    assert "Unsupported document type" in result.output
    assert ".pdf" in result.output and ".md" in result.output


def test_unsupported_type_can_be_forced_to_text(tmp_path):
    """--force-text is the escape hatch so odd files are still usable."""
    doc = tmp_path / "notes.rst"
    doc.write_text("Title\n-----\nRST content about attention.", encoding="utf-8")

    with patch("docpulse.cli.ICAGatewayClient") as client_cls:
        mock = MagicMock()
        mock.chat_complete.return_value = LLMResponse(content="### Executive Summary\nForced text run.", model="m")
        client_cls.return_value = mock

        result = runner.invoke(app, ["--force-text", "analyze", str(doc)])

    assert result.exit_code == 0
    assert "Forced text run" in result.output


def test_missing_api_key_warns_but_does_not_block(tmp_path, monkeypatch):
    """Keyless gateways are legitimate, so a missing key is a note, not an error."""
    config = tmp_path / "keyless.json"
    config.write_text(json.dumps({"endpoint_url": "https://gateway.example.com/v1", "api_key": ""}), encoding="utf-8")
    monkeypatch.setenv("DOCPULSE_CONFIG_PATH", str(config))

    with patch("docpulse.cli.ICAGatewayClient") as client_cls:
        mock = MagicMock()
        mock.chat_complete.return_value = LLMResponse(content="### Summary\nWorks.", model="m")
        client_cls.return_value = mock

        result = runner.invoke(app, ["analyze", str(write_doc(tmp_path))])

    assert result.exit_code == 0
    assert "No API key" in result.output
    assert "Works." in result.output


def test_empty_document_exits_4_instead_of_querying_the_model(tmp_path):
    empty = tmp_path / "empty.md"
    empty.write_text("", encoding="utf-8")

    with patch("docpulse.cli.ICAGatewayClient") as client_cls:
        mock = MagicMock()
        client_cls.return_value = mock

        result = runner.invoke(app, ["analyze", str(empty)])

    assert result.exit_code == EXIT_DOCUMENT
    assert "no extractable text" in result.output
    mock.chat_complete.assert_not_called()


def test_directory_instead_of_document_exits_4(tmp_path):
    result = runner.invoke(app, ["analyze", str(tmp_path)])

    assert result.exit_code == EXIT_DOCUMENT
    assert "directory" in result.output


def test_unknown_section_id_exits_4(tmp_path):
    doc = write_doc(tmp_path)

    with patch("docpulse.cli.ICAGatewayClient"):
        result = runner.invoke(app, ["section", str(doc), "--id", "99"])

    assert result.exit_code == EXIT_DOCUMENT
    assert "not found" in result.output


# --- Truncated and unusable model responses ---------------------------------


@patch("docpulse.cli.ICAGatewayClient")
def test_truncated_response_warns_instead_of_looking_empty(mock_client_cls, tmp_path):
    doc = write_doc(tmp_path)

    mock = MagicMock()
    mock.chat_complete.return_value = LLMResponse(
        content='{"concepts": [{"name": "truncated ha',
        model="m",
        finish_reason="length",
    )
    mock_client_cls.return_value = mock

    result = runner.invoke(app, ["concepts", str(doc)])

    assert result.exit_code == 0
    assert "token limit" in result.output
    assert "max_tokens" in result.output


@patch("docpulse.cli.ICAGatewayClient")
def test_truncated_response_is_reported_in_json_output(mock_client_cls, tmp_path):
    doc = write_doc(tmp_path)

    mock = MagicMock()
    mock.chat_complete.return_value = LLMResponse(content="{partial", model="m", finish_reason="length")
    mock_client_cls.return_value = mock

    result = runner.invoke(app, ["concepts", str(doc), "--format", "json"])

    assert result.exit_code == 0
    assert any("token limit" in w for w in json.loads(result.output)["warnings"])


@patch("docpulse.cli.ICAGatewayClient")
def test_unparseable_json_is_warned_about_rather_than_reported_as_no_concepts(mock_client_cls, tmp_path):
    doc = write_doc(tmp_path)

    mock = MagicMock()
    mock.chat_complete.return_value = LLMResponse(content="I'm afraid I can't help with that.", model="m")
    mock_client_cls.return_value = mock

    result = runner.invoke(app, ["concepts", str(doc)])

    assert result.exit_code == 0
    assert "not parseable JSON" in result.output


@patch("docpulse.cli.ICAGatewayClient")
def test_gateway_failure_exits_3(mock_client_cls, tmp_path):
    from docpulse.errors import LLMError

    doc = write_doc(tmp_path)

    mock = MagicMock()
    mock.chat_complete.side_effect = LLMError("Could not reach the gateway.")
    mock_client_cls.return_value = mock

    result = runner.invoke(app, ["analyze", str(doc)])

    assert result.exit_code == EXIT_GATEWAY
    assert "Could not reach the gateway" in result.output


@patch("docpulse.cli.ICAGatewayClient")
def test_unexpected_errors_hide_details_unless_debug_is_passed(mock_client_cls, tmp_path):
    doc = write_doc(tmp_path)

    mock = MagicMock()
    mock.chat_complete.side_effect = ZeroDivisionError("synthetic blowup")
    mock_client_cls.return_value = mock

    quiet = runner.invoke(app, ["analyze", str(doc)])
    assert quiet.exit_code == 1
    assert "--debug" in quiet.output

    loud = runner.invoke(app, ["--debug", "analyze", str(doc)])
    assert loud.exit_code == 1
    assert "synthetic blowup" in loud.output


# --- Global flags -----------------------------------------------------------


def test_help_lists_the_global_flags():
    result = runner.invoke(app, ["--help"])

    assert "--debug" in result.output
    assert "--force-text" in result.output


@pytest.mark.parametrize("flag", ["--debug", "--force-text"])
def test_global_flags_are_accepted_by_commands(tmp_path, monkeypatch, flag):
    monkeypatch.setenv("DOCPULSE_CONFIG_PATH", str(tmp_path / "missing.json"))
    doc = write_doc(tmp_path)

    result = runner.invoke(app, [flag, "analyze", str(doc)])

    # Config failure, not "no such option".
    assert "No such option" not in result.output
    assert result.exit_code == EXIT_USAGE