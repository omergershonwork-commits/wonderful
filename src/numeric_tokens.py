"""Shared complete-token numeric parsing for user and public-data boundaries."""
from __future__ import annotations

import re
from collections.abc import Iterable

# Capture the complete numeric-like token, including forms that must be rejected
# later: exponents, underscores, ranges, decimals, and malformed separators.
NUMERIC_TOKEN_PATTERN = r"(?P<token>[+-]?\d(?:[\dA-Za-z_]|[.,+-](?=\d))*)"

_CANONICAL_INTEGER = re.compile(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)")
_CANONICAL_DECIMAL = re.compile(
    r"[+-]?(?:(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?)"
)


class NumericTokenError(ValueError):
    """Raised when an explicitly supplied numeric token is unsupported."""


def find_numeric_token(text: str, patterns: Iterable[str]) -> str | None:
    """Return the first complete numeric-like token matched by a context pattern.

    Each pattern must contain ``{token}``, which is replaced by
    :data:`NUMERIC_TOKEN_PATTERN`.
    """

    lowered = text.casefold()
    for template in patterns:
        match = re.search(template.format(token=NUMERIC_TOKEN_PATTERN), lowered)
        if match:
            return match.group("token")
    return None


def parse_canonical_integer(token: str, *, field_name: str) -> int:
    """Parse an optionally signed integer with standard thousands grouping."""

    if _CANONICAL_INTEGER.fullmatch(token) is None:
        raise NumericTokenError(
            f"{field_name} must use canonical whole numbers with optional sign "
            "and correctly grouped commas"
        )
    return int(token.replace(",", ""))


def parse_canonical_decimal(token: str, *, field_name: str) -> float:
    """Parse a decimal without exponent, underscore, range, or bad grouping."""

    if _CANONICAL_DECIMAL.fullmatch(token) is None:
        raise NumericTokenError(
            f"{field_name} must use a canonical decimal number with optional sign, "
            "a decimal point, and correctly grouped commas"
        )
    return float(token.replace(",", ""))
