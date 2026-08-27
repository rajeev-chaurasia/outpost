"""Sanity checks for the shared types other phases build on."""

from outpost.agent import Citation, Rung
from outpost.mapping import MappingOutcome
from outpost.retrieval import Span


def test_span_holds_a_source_range() -> None:
    span = Span(source_id="statements", document_id="doc-1", start=10, end=20, text="short paid")
    assert span.end - span.start == len(span.text)


def test_citation_binds_a_claim_to_a_span() -> None:
    span = Span(source_id="statements", document_id="doc-1", start=0, end=5, text="hello")
    citation = Citation(assertion="the balance is off", span=span)
    assert citation.span == span


def test_mapping_outcome_values() -> None:
    assert {outcome.value for outcome in MappingOutcome} == {"mapped", "needs_review", "unmapped"}


def test_rung_ordering() -> None:
    assert list(Rung) == sorted(Rung)
    assert Rung.FULL < Rung.REFUSED
