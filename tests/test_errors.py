"""Tests for the typed error hierarchy and its exit-code contract."""

import pytest

from docpulse.errors import (
    EXIT_DOCUMENT,
    EXIT_GATEWAY,
    EXIT_INTERNAL,
    EXIT_NO_RESULT,
    EXIT_USAGE,
    ConfigError,
    DocPulseError,
    DocumentError,
    EmptyResultError,
    LLMError,
    TruncatedResponseError,
    UsageError,
)


@pytest.mark.parametrize(
    ("error_type", "expected_code"),
    [
        (ConfigError, EXIT_USAGE),
        (UsageError, EXIT_USAGE),
        (LLMError, EXIT_GATEWAY),
        (TruncatedResponseError, EXIT_GATEWAY),
        (DocumentError, EXIT_DOCUMENT),
        (EmptyResultError, EXIT_NO_RESULT),
        (DocPulseError, EXIT_INTERNAL),
    ],
)
def test_each_error_type_carries_its_documented_exit_code(error_type, expected_code):
    assert error_type("boom").exit_code == expected_code


def test_truncated_response_is_a_gateway_error_not_a_document_error():
    assert issubclass(TruncatedResponseError, LLMError)


def test_user_message_appends_the_hint():
    error = LLMError("Could not reach the gateway.", hint="Check the URL.")
    assert "Could not reach the gateway." in error.user_message()
    assert "Check the URL." in error.user_message()


def test_user_message_without_hint_is_just_the_message():
    assert LLMError("Plain failure.").user_message() == "Plain failure."


def test_error_codes_are_distinct():
    codes = {EXIT_INTERNAL, EXIT_USAGE, EXIT_GATEWAY, EXIT_DOCUMENT, EXIT_NO_RESULT}
    assert len(codes) == 5