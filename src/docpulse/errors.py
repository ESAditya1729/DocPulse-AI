"""Typed error hierarchy for DocPulse with stable process exit codes.

Every failure DocPulse can anticipate is raised as a `DocPulseError` subclass
carrying the exit code the CLI should terminate with. This lets the CLI print
an actionable message instead of "Error communicating with LLM Gateway" for
every distinct failure mode, and lets scripts branch on the exit code instead
of grepping stdout.

Exit code contract:
    0  success
    1  unexpected internal error (traceback shown only with --debug)
    2  usage or configuration error (bad flag, DocPulse not initialized)
    3  gateway failure (network, timeout, HTTP status, truncated response)
    4  document error (missing, unsupported type, empty, unreadable)
    5  model returned nothing usable (unparseable or empty output)
"""

# --- Exit codes ------------------------------------------------------------

EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_USAGE = 2
EXIT_GATEWAY = 3
EXIT_DOCUMENT = 4
EXIT_NO_RESULT = 5

# --- Error types -----------------------------------------------------------


class DocPulseError(Exception):
    """Base class for every anticipated DocPulse failure."""

    exit_code: int = EXIT_INTERNAL

    def __init__(self, message: str, *, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def user_message(self) -> str:
        """Message plus an optional recovery hint, as shown to the user."""
        if self.hint:
            return f"{self.message}\n\n{self.hint}"
        return self.message


class ConfigError(DocPulseError):
    """DocPulse is not configured, or the gateway rejected our credentials."""

    exit_code = EXIT_USAGE


class UsageError(DocPulseError):
    """The command was invoked with invalid or conflicting arguments."""

    exit_code = EXIT_USAGE


class LLMError(DocPulseError):
    """The gateway could not be reached or returned an error response."""

    exit_code = EXIT_GATEWAY


class TruncatedResponseError(LLMError):
    """The model stopped mid-response because it hit the token limit."""

    exit_code = EXIT_GATEWAY


class DocumentError(DocPulseError):
    """The input document is missing, unsupported, empty, or unreadable."""

    exit_code = EXIT_DOCUMENT


class EmptyResultError(DocPulseError):
    """The model responded, but nothing usable could be extracted from it."""

    exit_code = EXIT_NO_RESULT