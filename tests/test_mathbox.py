"""Tests for math equation extraction and Unicode prettification."""

from docpulse.mathbox import extract_equations, prettify_math, split_segments


def test_extract_inline_and_block_equations():
    text = (
        "A recurrence relation $A^{n-m}$ is derived for simplified exponentiation "
        "using the diagonalized form, $A^{n-m} = Λ(γe^{iθ})^{n-m}Λ^{-1}$.\n\n"
        "$$o_n = \\sum_{m=1}^n \\gamma^{n-m}$$"
    )
    equations = extract_equations(text)

    assert "A^{n-m}" in equations
    assert any(eq.startswith("A^{n-m} =") for eq in equations)
    assert "o_n = \\sum_{m=1}^n \\gamma^{n-m}" in equations


def test_extract_skips_price_like_dollar_amounts():
    text = "The license costs $5 and renews for $10 each year."
    assert extract_equations(text) == []


def test_extract_deduplicates_repeated_equations():
    text = "First mention: $E=mc^2$. Later again: $E=mc^2$."
    assert extract_equations(text) == ["E=mc^2"]


def test_prettify_converts_greek_letters_and_scripts():
    pretty = prettify_math(r"\gamma^{n-m}")
    assert pretty == "γⁿ⁻m"  # no Unicode superscript glyph exists for 'm', so it stays plain


def test_prettify_handles_sum_and_subscripts():
    pretty = prettify_math(r"\sum_{m=1}^n \gamma^{n-m}")
    assert "Σ" in pretty
    assert "γ" in pretty
    assert "{" not in pretty and "}" not in pretty
    assert "\\sum" not in pretty and "\\gamma" not in pretty


def test_prettify_strips_latex_delimiters_and_spacing_commands():
    pretty = prettify_math(r"\left( \gamma \right)")
    assert "\\left" not in pretty
    assert "\\right" not in pretty


def test_prettify_drops_orphan_caret_before_symbol_but_keeps_it_before_letters():
    # "\dagger" has no superscript glyph, so "^\dagger" would otherwise leave
    # a dangling "^†" - the caret should be dropped, keeping just the symbol.
    assert prettify_math(r"v_m)^\dagger") == "vₘ)†"
    # A caret before an ordinary letter is still a recognizable ASCII
    # exponent marker and should be left as-is.
    assert prettify_math(r"x^T") == "x^T"


def test_split_segments_flattens_inline_math_into_surrounding_text():
    text = "- Λ is absorbed into $W_Q$ and $W_K$ to ease computation."
    segments = split_segments(text)

    # No separate "math" segment for inline equations: everything, including
    # the flattened inline math, stays inside one "text" chunk so the bullet
    # marker and sentence flow survive intact.
    assert [kind for kind, _ in segments] == ["text"]
    chunk = segments[0][1]
    assert chunk.startswith("- ")
    assert "W_Q" in chunk and "W_K" in chunk
    assert "$" not in chunk


def test_split_segments_boxes_only_standalone_block_equations():
    text = "The recurrence is defined as:\n\n$$o_n = \\sum_{m=1}^n \\gamma^{n-m}$$\n\nThat's the full form."
    segments = split_segments(text)
    kinds = [kind for kind, _ in segments]

    assert kinds == ["text", "math", "text"]
    assert segments[1][1] == "o_n = \\sum_{m=1}^n \\gamma^{n-m}"


def test_split_segments_leaves_price_like_dollar_amounts_untouched():
    text = "The license costs $5 and renews for $10 each year."
    segments = split_segments(text)
    assert [kind for kind, _ in segments] == ["text"]
    assert segments[0][1] == text
