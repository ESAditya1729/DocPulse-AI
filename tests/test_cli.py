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
def test_cli_analyze_large_document_is_chunked_not_truncated(mock_client_cls, tmp_path, monkeypatch):
    monkeypatch.setenv("DOCPULSE_CONFIG_PATH", str(tmp_path / "config.json"))
    sample_file = tmp_path / "big.md"
    sample_file.write_text("# Title\n\n" + ("word " * 5000), encoding="utf-8")  # ~25000 chars, over the 16000 cutoff

    def fake_chat_complete(*, system_prompt=None, messages):
        content = messages[0]["content"]
        if "condensing part" in content:
            return LLMResponse(content="condensed notes", model="m")
        return LLMResponse(content="### Executive Summary\nFull coverage summary.", model="m")

    mock_client = MagicMock()
    mock_client.chat_complete.side_effect = fake_chat_complete
    mock_client_cls.return_value = mock_client

    result = runner.invoke(app, ["analyze", str(sample_file)])

    assert result.exit_code == 0
    assert "Full coverage summary" in result.output
    assert "condensed from" in result.output
    assert mock_client.chat_complete.call_count > 1


@patch("docpulse.cli.ICAGatewayClient")
def test_cli_analyze_second_run_served_from_cache(mock_client_cls, tmp_path, monkeypatch):
    monkeypatch.setenv("DOCPULSE_CONFIG_PATH", str(tmp_path / "config.json"))
    sample_file = tmp_path / "test.md"
    sample_file.write_text("# Section 1\nHello World", encoding="utf-8")

    mock_client = MagicMock()
    mock_client.chat_complete.return_value = LLMResponse(content="### Executive Summary\nCached summary.", model="m")
    mock_client_cls.return_value = mock_client

    first = runner.invoke(app, ["analyze", str(sample_file)])
    second = runner.invoke(app, ["analyze", str(sample_file)])

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert "Served from cache" not in first.output
    assert "Served from cache" in second.output
    mock_client.chat_complete.assert_called_once()


@patch("docpulse.cli.ICAGatewayClient")
def test_cli_ask_one_shot(mock_client_cls, tmp_path, monkeypatch):
    monkeypatch.setenv("DOCPULSE_CONFIG_PATH", str(tmp_path / "config.json"))
    sample_file = tmp_path / "test.md"
    sample_file.write_text("# Section 1\nDocPulse uses Typer and Rich.", encoding="utf-8")

    mock_client = MagicMock()
    mock_client.chat_complete.return_value = LLMResponse(content="It uses Typer and Rich.", model="m")
    mock_client_cls.return_value = mock_client

    result = runner.invoke(app, ["ask", str(sample_file), "What libraries does it use?"])

    assert result.exit_code == 0
    assert "It uses Typer and Rich." in result.output
    mock_client.chat_complete.assert_called_once()


@patch("docpulse.cli.ICAGatewayClient")
def test_cli_ask_interactive_session(mock_client_cls, tmp_path, monkeypatch):
    monkeypatch.setenv("DOCPULSE_CONFIG_PATH", str(tmp_path / "config.json"))
    sample_file = tmp_path / "test.md"
    sample_file.write_text("# Section 1\nDocPulse uses Typer and Rich.", encoding="utf-8")

    mock_client = MagicMock()
    mock_client.chat_complete.return_value = LLMResponse(content="Answer text.", model="m")
    mock_client_cls.return_value = mock_client

    result = runner.invoke(app, ["ask", str(sample_file)], input="What libraries does it use?\nexit\n")

    assert result.exit_code == 0
    assert "Answer text." in result.output
    assert "Ask questions about" in result.output
    mock_client.chat_complete.assert_called_once()


@patch("docpulse.cli.ICAGatewayClient")
def test_cli_init_command(mock_client_cls, tmp_path, monkeypatch):
    custom_cfg_path = tmp_path / "config.json"
    monkeypatch.setenv("DOCPULSE_CONFIG_PATH", str(custom_cfg_path))

    mock_client = MagicMock()
    mock_client.check_health.return_value = {
        "status": "ok",
        "endpoint": "https://gateway.example.com/v1/chat-models/models",
        "code": 200,
        "models_count": 5,
    }
    mock_client_cls.return_value = mock_client

    # Interactive inputs: endpoint, api_key, namespace, model_id
    inputs = "https://gateway.example.com/v1\ntest-api-key\nchat-models\nibm/granite-3-8b-instruct\n"
    result = runner.invoke(app, ["init"], input=inputs)
    assert result.exit_code == 0
    assert "Configuration saved" in result.output
    assert "Connection successful!" in result.output


@patch("docpulse.cli.ICAGatewayClient")
def test_cli_init_warns_on_unknown_model_but_allows_override(mock_client_cls, tmp_path, monkeypatch):
    monkeypatch.setenv("DOCPULSE_CONFIG_PATH", str(tmp_path / "config.json"))

    mock_client = MagicMock()
    mock_client.list_models.return_value = [{"id": "known-model-a"}, {"id": "known-model-b"}]
    mock_client.check_health.return_value = {"status": "ok", "endpoint": "https://gateway.example.com/v1", "code": 200}
    mock_client_cls.return_value = mock_client

    # endpoint, api_key, namespace, model_id (typo'd/unknown), confirm "use it anyway" = yes
    inputs = "https://gateway.example.com/v1\ntest-api-key\nchat-models\ntypo-model\ny\n"
    result = runner.invoke(app, ["init"], input=inputs)

    assert result.exit_code == 0
    assert "wasn't in the models discovered" in result.output
    assert "Configuration saved" in result.output


@patch("docpulse.cli.ICAGatewayClient")
def test_cli_init_accepts_known_model_without_warning(mock_client_cls, tmp_path, monkeypatch):
    monkeypatch.setenv("DOCPULSE_CONFIG_PATH", str(tmp_path / "config.json"))

    mock_client = MagicMock()
    mock_client.list_models.return_value = [{"id": "known-model-a"}, {"id": "known-model-b"}]
    mock_client.check_health.return_value = {"status": "ok", "endpoint": "https://gateway.example.com/v1", "code": 200}
    mock_client_cls.return_value = mock_client

    inputs = "https://gateway.example.com/v1\ntest-api-key\nchat-models\nknown-model-a\n"
    result = runner.invoke(app, ["init"], input=inputs)

    assert result.exit_code == 0
    assert "wasn't in the models discovered" not in result.output
    assert "Configuration saved" in result.output
