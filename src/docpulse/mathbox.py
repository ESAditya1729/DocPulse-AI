"""Extraction and best-effort Unicode prettification of inline/display math.

The LLM is instructed (see prompts.SYSTEM_PROMPT) to prefer plain Unicode math
and fall back to $...$ / $$...$$ delimiters only when necessary. This module
splits LLM markdown output into an ordered sequence of plain-text and math
segments, in their original position, so the CLI can render each equation as
its own boxed element embedded exactly where it occurs - instead of pulling
every equation out to a separate section - and cleans up any residual LaTeX
commands the model used inside them (terminals have no LaTeX/KaTeX renderer).
"""

import re

# Tries the $$...$$ (block) alternative first at each position; falls back to
# $...$ (inline) only when a double-dollar match isn't possible there.
_SEGMENT_RE = re.compile(r"\$\$(?P<block>.+?)\$\$|\$(?P<inline>[^$\n]+?)\$", re.DOTALL)

# Heuristic: an inline $...$ match only counts as math (as opposed to a price
# like "$5") if it contains at least one of these math-ish characters.
_MATH_HINT_CHARS = set("\\^_={}")

_LATEX_SYMBOLS = {
    r"\sum": "Σ", r"\prod": "∏", r"\int": "∫", r"\sqrt": "√",
    r"\cdot": "·", r"\times": "×", r"\pm": "±", r"\mp": "∓",
    r"\leq": "≤", r"\geq": "≥", r"\neq": "≠", r"\approx": "≈",
    r"\infty": "∞", r"\partial": "∂", r"\nabla": "∇", r"\dagger": "†",
    r"\to": "→", r"\rightarrow": "→", r"\leftarrow": "←",
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\epsilon": "ε", r"\zeta": "ζ", r"\eta": "η", r"\theta": "θ",
    r"\iota": "ι", r"\kappa": "κ", r"\lambda": "λ", r"\mu": "μ",
    r"\nu": "ν", r"\xi": "ξ", r"\pi": "π", r"\rho": "ρ",
    r"\sigma": "σ", r"\tau": "τ", r"\upsilon": "υ", r"\phi": "φ",
    r"\chi": "χ", r"\psi": "ψ", r"\omega": "ω",
    r"\Gamma": "Γ", r"\Delta": "Δ", r"\Theta": "Θ", r"\Lambda": "Λ",
    r"\Xi": "Ξ", r"\Pi": "Π", r"\Sigma": "Σ", r"\Phi": "Φ",
    r"\Psi": "Ψ", r"\Omega": "Ω",
}

_SUPERSCRIPT = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶",
    "7": "⁷", "8": "⁸", "9": "⁹", "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽",
    ")": "⁾", "n": "ⁿ", "i": "ⁱ",
}
_SUBSCRIPT = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆",
    "7": "₇", "8": "₈", "9": "₉", "+": "₊", "-": "₋", "=": "₌", "(": "₍",
    ")": "₎", "a": "ₐ", "e": "ₑ", "i": "ᵢ", "j": "ⱼ", "k": "ₖ", "m": "ₘ",
    "n": "ₙ", "o": "ₒ", "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ", "u": "ᵤ",
    "v": "ᵥ", "x": "ₓ",
}


def _transliterate_script(text: str, braced: re.Pattern, single: re.Pattern, table: dict[str, str]) -> str:
    def repl(match: re.Match) -> str:
        body = match.group(1)
        converted = "".join(table.get(ch, ch) for ch in body)
        return converted if converted != body else match.group(0)

    text = braced.sub(repl, text)
    text = single.sub(repl, text)
    return text


_SUPER_BRACED = re.compile(r"\^\{([^{}]+)\}")
_SUPER_SINGLE = re.compile(r"\^(\w)")
_SUB_BRACED = re.compile(r"_\{([^{}]+)\}")
_SUB_SINGLE = re.compile(r"_(\w)")


def prettify_math(raw: str) -> str:
    """Best-effort conversion of residual LaTeX commands to Unicode for display."""
    text = raw.strip()

    for latex, glyph in _LATEX_SYMBOLS.items():
        text = text.replace(latex, glyph)

    text = _transliterate_script(text, _SUPER_BRACED, _SUPER_SINGLE, _SUPERSCRIPT)
    text = _transliterate_script(text, _SUB_BRACED, _SUB_SINGLE, _SUBSCRIPT)

    # A "^" immediately before a symbol/punctuation character (e.g.
    # "^\dagger" -> "^†": dagger isn't a \w char, so the superscript regex
    # above never touched it) can't be made into a real Unicode superscript -
    # drop the orphaned caret rather than leave it dangling. A caret before a
    # letter/digit (e.g. "x^T", left alone above because no superscript "T"
    # glyph exists) is kept: "^" is still a widely-understood ASCII
    # exponent marker there, unlike in front of a symbol.
    text = re.sub(r"\^(?=\W)", "", text)

    for noop in (r"\left", r"\right", r"\,", r"\;", r"\:", r"\!"):
        text = text.replace(noop, "")

    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


def _iter_matches(markdown_text: str):
    """Yield (kind, payload, start, end) for every $...$/$$...$$ match, in order.

    kind is "block" or "inline" for a genuine equation (payload = the equation
    text), or "literal" for a match that doesn't look like math - an empty
    $$$$ or a price-like "$5" - in which case payload is the raw matched text
    (delimiters included) so callers can put it back verbatim.
    """
    for match in _SEGMENT_RE.finditer(markdown_text):
        start, end = match.span()
        block = match.group("block")
        if block is not None:
            eq = block.strip()
            if eq:
                yield "block", eq, start, end
            else:
                yield "literal", match.group(0), start, end
            continue

        eq = (match.group("inline") or "").strip()
        if eq and any(ch in _MATH_HINT_CHARS for ch in eq):
            yield "inline", eq, start, end
        else:
            yield "literal", match.group(0), start, end


def extract_equations(markdown_text: str) -> list[str]:
    """Pull every math equation (inline and block) out, in order, de-duplicated."""
    equations = [payload for kind, payload, _, _ in _iter_matches(markdown_text) if kind in ("block", "inline")]

    seen: set[str] = set()
    unique: list[str] = []
    for eq in equations:
        if eq not in seen:
            seen.add(eq)
            unique.append(eq)
    return unique


def split_segments(markdown_text: str) -> list[tuple[str, str]]:
    """Split text into ordered ("text", chunk) / ("math", equation) segments.

    Only standalone $$...$$ block equations become their own ("math", ...)
    segment, so a caller can box each one in place. Inline $...$ equations
    are flattened into the surrounding text as prettified Unicode instead -
    boxing something sitting mid-sentence or mid-bullet would break the line
    it's part of. An inline $...$ span that doesn't look like real math
    (e.g. "$5") is left untouched as plain text.
    """
    segments: list[tuple[str, str]] = []
    pending: list[str] = []
    last_end = 0

    def flush_pending(gap: str) -> None:
        text = "".join(pending) + gap
        if text:
            segments.append(("text", text))
        pending.clear()

    for kind, payload, start, end in _iter_matches(markdown_text):
        gap = markdown_text[last_end:start]
        last_end = end

        if kind == "block":
            flush_pending(gap)
            segments.append(("math", payload))
        elif kind == "inline":
            pending.append(gap + prettify_math(payload))
        else:  # literal: not math, keep the original text as-is
            pending.append(gap + payload)

    flush_pending(markdown_text[last_end:])
    return segments
