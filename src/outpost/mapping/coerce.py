"""Date, currency, and identifier normalization.

Every coercion returns a CoerceResult instead of raising: a value that
cannot be read confidently (an ambiguous date, an empty field) comes back
as NEEDS_REVIEW rather than a guess, so it lands in the MappingReport
instead of silently becoming wrong data.
"""

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from outpost.mapping.report import MappingOutcome

FieldKind = Literal["date", "currency", "identifier", "text"]

_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_SLASH_DATE_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")


@dataclass(frozen=True)
class CoerceResult:
    value: Any
    outcome: MappingOutcome
    reason: str | None = None


def _is_missing(raw: str) -> bool:
    return not raw.strip() or raw.strip().upper() == "NULL"


def classify_field_kind(field_name: str) -> FieldKind:
    """Infers how to coerce a field purely from its name's shape (a
    "_date" suffix, an "amount" substring, an "_id" or "_number" suffix),
    never from a tenant's specific vocabulary, so this stays reusable
    across every tenant's ontology.
    """
    lowered = field_name.lower()
    if lowered.endswith("_date") or lowered == "date":
        return "date"
    if "amount" in lowered or lowered.endswith("_limit"):
        return "currency"
    if lowered.endswith("_id") or lowered.endswith("_number"):
        return "identifier"
    return "text"


def coerce_date(raw: str) -> CoerceResult:
    stripped = raw.strip()
    if _is_missing(raw):
        return CoerceResult(value=None, outcome=MappingOutcome.NEEDS_REVIEW, reason="missing value")

    iso_match = _ISO_DATE_RE.match(stripped)
    if iso_match:
        year, month, day = (int(group) for group in iso_match.groups())
        try:
            return CoerceResult(value=date(year, month, day), outcome=MappingOutcome.MAPPED)
        except ValueError:
            return CoerceResult(
                value=None, outcome=MappingOutcome.NEEDS_REVIEW, reason="invalid calendar date"
            )

    slash_match = _SLASH_DATE_RE.match(stripped)
    if slash_match:
        first, second, year = (int(group) for group in slash_match.groups())
        # Slash dates in these fixtures are day/month/year. If both parts
        # could plausibly be the month, the order can't be told apart
        # from the value alone, so this is flagged instead of guessed.
        if first <= 12 and second <= 12:
            return CoerceResult(
                value=None, outcome=MappingOutcome.NEEDS_REVIEW, reason="ambiguous date format"
            )
        day, month = (first, second) if first > 12 else (second, first)
        try:
            return CoerceResult(value=date(year, month, day), outcome=MappingOutcome.MAPPED)
        except ValueError:
            return CoerceResult(
                value=None, outcome=MappingOutcome.NEEDS_REVIEW, reason="invalid calendar date"
            )

    return CoerceResult(
        value=None, outcome=MappingOutcome.NEEDS_REVIEW, reason="unrecognized date format"
    )


def coerce_currency(raw: str) -> CoerceResult:
    if _is_missing(raw):
        return CoerceResult(value=None, outcome=MappingOutcome.NEEDS_REVIEW, reason="missing value")

    cleaned = raw.strip().replace("$", "").replace(",", "")
    try:
        return CoerceResult(value=Decimal(cleaned), outcome=MappingOutcome.MAPPED)
    except InvalidOperation:
        return CoerceResult(
            value=None, outcome=MappingOutcome.NEEDS_REVIEW, reason="unparseable amount"
        )


def coerce_identifier(raw: str) -> CoerceResult:
    if _is_missing(raw):
        return CoerceResult(value=None, outcome=MappingOutcome.NEEDS_REVIEW, reason="missing value")
    return CoerceResult(value=raw.strip(), outcome=MappingOutcome.MAPPED)


def coerce_text(raw: str) -> CoerceResult:
    if _is_missing(raw):
        return CoerceResult(value=None, outcome=MappingOutcome.NEEDS_REVIEW, reason="missing value")
    return CoerceResult(value=raw.strip(), outcome=MappingOutcome.MAPPED)


def coerce_value(kind: FieldKind, raw: str) -> CoerceResult:
    if kind == "date":
        return coerce_date(raw)
    if kind == "currency":
        return coerce_currency(raw)
    if kind == "identifier":
        return coerce_identifier(raw)
    return coerce_text(raw)
