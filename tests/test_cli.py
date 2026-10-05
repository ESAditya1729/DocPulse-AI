"""Tests for CLI command execution using Typer CliRunner."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from docpulse.cli import app
from docpulse.llm import LLMResponse

runner = CliRunner()


def test_cli_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "DocPulse version" in result.output


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "DocPulse" in result.output
    assert "analyze" in result.output
    assert "section" in result.output
    assert "sources" in result.output
    assert "export" in result.output
    assert "init" in result.output


@patch("docpulse.cli.ICAGatewayClient")
def test_cli_analyze_command(mock_client_cls, tmp_path):
    sample_file = tmp_path / "test.md"
    sample_file.write_text("# Section 1\nHello World", encoding="utf-8")

    mock_client = MagicMock()
    mock_client.chat_complete.return_value = LLMResponse(
        content="### Executive Summary\nDocument test summary.",
        model="ibm/granite-3-8b",
    )
    mock_client_cls.return_value = mock_client

    result = runner.invoke(app, ["analyze", str(sample_file)])
    assert result.exit_code == 0
    assert "Executive Summary" in result.output


@patch("docpulse.cli.ICAGatewayClient")
def test_cli_init_command(mock_client_cls, tmp_path, monkeypatch):
    custom_cfg_path = tmp_path / "config.json"
    monkeypatch.setenv("DOCPULSE_CONFIG_PATH", str(custom_cfg_path))

    mock_client = MagicMock()
    mock_client.check_health.return_value = {
        "status": "ok",
        "endpoint": "https://api.nextgen-beta.ica.ibm.com/ica/v1/chat-models/models",
        "code": 200,
        "models_count": 5,
    }
    mock_client_cls.return_value = mock_client

    # Interactive inputs: endpoint, api_key, namespace, model_id
    inputs = "https://api.nextgen-beta.ica.ibm.com/ica/v1\ntest-api-key\nchat-models\nibm/granite-3-8b-instruct\n"
    result = runner.invoke(app, ["init"], input=inputs)
    assert result.exit_code == 0
    assert "Configuration saved" in result.output
    assert "Connection successful!" in result.output
