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
    assert "concepts" in result.output
    assert "prerequisites" in result.output
    assert "equations" in result.output
    assert "map" in result.output
    assert "compare" in result.output
    assert "study" in result.output
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
def test_cli_analyze_large_document_is_chunked_not_truncated(mock_client_cls, tmp_path):
    
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
def test_cli_analyze_second_run_served_from_cache(mock_client_cls, tmp_path):
    
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
def test_cli_ask_one_shot(mock_client_cls, tmp_path):
    
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
def test_cli_ask_interactive_session(mock_client_cls, tmp_path):
    
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
def test_cli_init_command(mock_client_cls, tmp_path):
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
def test_cli_init_warns_on_unknown_model_but_allows_override(mock_client_cls, tmp_path):
    

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
def test_cli_init_accepts_known_model_without_warning(mock_client_cls, tmp_path):
    

    mock_client = MagicMock()
    mock_client.list_models.return_value = [{"id": "known-model-a"}, {"id": "known-model-b"}]
    mock_client.check_health.return_value = {"status": "ok", "endpoint": "https://gateway.example.com/v1", "code": 200}
    mock_client_cls.return_value = mock_client

    inputs = "https://gateway.example.com/v1\ntest-api-key\nchat-models\nknown-model-a\n"
    result = runner.invoke(app, ["init"], input=inputs)

    assert result.exit_code == 0
    assert "wasn't in the models discovered" not in result.output
    assert "Configuration saved" in result.output


@patch("docpulse.cli.ICAGatewayClient")
def test_cli_concepts_command_text_and_json(mock_client_cls, tmp_path):
    
    sample_file = tmp_path / "test.md"
    sample_file.write_text("# Attention Is All You Need\nExplaining multi-head attention.", encoding="utf-8")

    mock_client = MagicMock()
    mock_client.chat_complete.return_value = LLMResponse(
        content='{"concepts": [{"name": "Multi-Head Attention", "importance": "high", "description": "Attends across subspaces", "sections": [1]}]}',
        model="m",
    )
    mock_client_cls.return_value = mock_client

    # Text mode
    res_text = runner.invoke(app, ["concepts", str(sample_file)])
    assert res_text.exit_code == 0
    assert "Multi-Head Attention" in res_text.output

    # JSON mode
    res_json = runner.invoke(app, ["concepts", str(sample_file), "--format", "json"])
    assert res_json.exit_code == 0
    assert '"name": "Multi-Head Attention"' in res_json.output

    # Name filter
    res_filtered = runner.invoke(app, ["concepts", str(sample_file), "--name", "attention"])
    assert res_filtered.exit_code == 0
    assert "Multi-Head Attention" in res_filtered.output


@patch("docpulse.cli.ICAGatewayClient")
def test_cli_prerequisites_command(mock_client_cls, tmp_path):
    
    sample_file = tmp_path / "test.md"
    sample_file.write_text("# Math Paper\nRequires linear algebra.", encoding="utf-8")

    mock_client = MagicMock()
    mock_client.chat_complete.return_value = LLMResponse(
        content='{"prerequisites": [{"name": "Linear Algebra", "importance": "high", "difficulty": "medium", "needed_for": "Matrix calculus"}]}',
        model="m",
    )
    mock_client_cls.return_value = mock_client

    res = runner.invoke(app, ["prerequisites", str(sample_file)])
    assert res.exit_code == 0
    assert "Linear Algebra" in res.output
    assert "Matrix calculus" in res.output


@patch("docpulse.cli.ICAGatewayClient")
def test_cli_equations_command(mock_client_cls, tmp_path):
    
    sample_file = tmp_path / "test.md"
    sample_file.write_text("# Formula\nE = mc^2", encoding="utf-8")

    mock_client = MagicMock()
    mock_client.chat_complete.return_value = LLMResponse(
        content='{"equations": [{"id": 1, "latex": "E = mc^2", "readable": "E = mc²", "section": "1", "variables": [{"symbol": "E", "description": "Energy"}], "explanation": "Mass-energy equivalence"}]}',
        model="m",
    )
    mock_client_cls.return_value = mock_client

    res = runner.invoke(app, ["equations", str(sample_file)])
    assert res.exit_code == 0
    assert "Equation 1" in res.output
    assert "Mass-energy equivalence" in res.output


@patch("docpulse.cli.ICAGatewayClient")
def test_cli_map_command_text_json_mermaid(mock_client_cls, tmp_path):
    
    sample_file = tmp_path / "test.md"
    sample_file.write_text("# Overview\nExplains architecture.", encoding="utf-8")

    mock_client = MagicMock()
    mock_client.chat_complete.side_effect = [
        LLMResponse(content='{"concepts": [{"name": "Transformer", "importance": "high"}]}', model="m"),
        LLMResponse(content='{"prerequisites": [{"name": "Calculus", "importance": "medium"}]}', model="m"),
        LLMResponse(content='{"equations": []}', model="m"),
        LLMResponse(content='{"references": ["Attention Is All You Need (arXiv:1706.03762)"]}', model="m"),
    ]
    mock_client_cls.return_value = mock_client

    # Text mode
    res_text = runner.invoke(app, ["map", str(sample_file)])
    assert res_text.exit_code == 0
    assert "DOCUMENT MAP" in res_text.output
    assert "Transformer" in res_text.output
    assert "References & Sources" in res_text.output
    assert "Attention Is All You Need" in res_text.output

    # JSON mode
    mock_client.chat_complete.side_effect = [
        LLMResponse(content='{"concepts": [{"name": "Transformer", "importance": "high"}]}', model="m"),
        LLMResponse(content='{"prerequisites": [{"name": "Calculus", "importance": "medium"}]}', model="m"),
        LLMResponse(content='{"equations": []}', model="m"),
        LLMResponse(content='{"references": ["Attention Is All You Need (arXiv:1706.03762)"]}', model="m"),
    ]
    res_json = runner.invoke(app, ["map", str(sample_file), "--format", "json"])
    assert res_json.exit_code == 0
    assert '"core_concepts"' in res_json.output
    assert "Attention Is All You Need" in res_json.output

    # Mermaid mode
    mock_client.chat_complete.side_effect = [
        LLMResponse(content='{"concepts": [{"name": "Transformer", "importance": "high"}]}', model="m"),
        LLMResponse(content='{"prerequisites": [{"name": "Calculus", "importance": "medium"}]}', model="m"),
        LLMResponse(content='{"equations": []}', model="m"),
        LLMResponse(content='{"references": ["Attention Is All You Need (arXiv:1706.03762)"]}', model="m"),
    ]
    res_mermaid = runner.invoke(app, ["map", str(sample_file), "--format", "mermaid"])
    assert res_mermaid.exit_code == 0
    assert "graph TD" in res_mermaid.output
    assert "subgraph References" in res_mermaid.output


@patch("docpulse.cli.ICAGatewayClient")
def test_cli_compare_command(mock_client_cls, tmp_path):
    
    f1 = tmp_path / "doc1.md"
    f2 = tmp_path / "doc2.md"
    f1.write_text("# Doc 1\nUses RNNs.", encoding="utf-8")
    f2.write_text("# Doc 2\nUses Transformers.", encoding="utf-8")

    mock_client = MagicMock()
    mock_client.chat_complete.return_value = LLMResponse(
        content='{"shared_concepts": ["Sequence Modeling"], "unique_to_a": ["Recurrence"], "unique_to_b": ["Self-Attention"], "key_differences": ["Parallelism vs Sequential"], "conclusion": "Doc 2 enables parallel training."}',
        model="m",
    )
    mock_client_cls.return_value = mock_client

    res = runner.invoke(app, ["compare", str(f1), str(f2)])
    assert res.exit_code == 0
    assert "DOCUMENT COMPARISON" in res.output
    assert "Sequence Modeling" in res.output
    assert "Self-Attention" in res.output


@patch("docpulse.cli.ICAGatewayClient")
def test_cli_study_command(mock_client_cls, tmp_path):
    
    sample_file = tmp_path / "test.md"
    sample_file.write_text("# Doc\nLearning content.", encoding="utf-8")

    mock_client = MagicMock()
    mock_client.chat_complete.return_value = LLMResponse(
        content='{"key_concepts": ["Attention"], "prerequisites": ["Vectors"], "important_equations": [], "flashcards": [{"question": "What is Q?", "answer": "Query"}], "questions": [{"question": "Why attention?", "type": "conceptual", "answer": "Better context"}]}',
        model="m",
    )
    mock_client_cls.return_value = mock_client

    res = runner.invoke(app, ["study", str(sample_file), "--questions", "3"])
    assert res.exit_code == 0
    assert "STUDY MODE" in res.output
    assert "Flashcards" in res.output
    assert "What is Q?" in res.output


@patch("docpulse.cli.ICAGatewayClient")
def test_cli_ask_with_citations(mock_client_cls, tmp_path):
    
    sample_file = tmp_path / "test.md"
    sample_file.write_text("# Section 1\nDocPulse supports citations.", encoding="utf-8")

    mock_client = MagicMock()
    mock_client.chat_complete.return_value = LLMResponse(
        content='{"answer": "It provides grounded citations.", "evidence": [{"section": "1", "section_title": "Section 1", "excerpt": "DocPulse supports citations"}]}',
        model="m",
    )
    mock_client_cls.return_value = mock_client

    res = runner.invoke(app, ["ask", str(sample_file), "Does it support citations?", "--citations"])
    assert res.exit_code == 0
    assert "It provides grounded citations." in res.output
    assert "Evidence & Citations" in res.output
    assert "Section: 1" in res.output
