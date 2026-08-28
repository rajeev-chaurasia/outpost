"""Grounding done-tests: a claim supported by retrieved evidence gets a
citation whose span resolves to real text, an unsupported claim is
counted rather than silently dropped or accepted, and every citation's
span text is exactly what the source span carries.
"""

from outpost.agent.ground import ground_answer
from outpost.retrieval.document import Span


def _span(text: str) -> Span:
    return Span(source_id="s", document_id="d", start=0, end=len(text), text=text)


def test_supported_claim_gets_a_citation() -> None:
    evidence = [_span("invoice INV-1001 was paid in full on 2026-03-15 via ach")]
    result = ground_answer("Invoice INV-1001 was paid in full on 2026-03-15 via ACH.", evidence)

    assert len(result.citations) == 1
    assert result.unsupported_assertions == []
    assert result.citations[0].span.text == evidence[0].text


def test_unrelated_claim_is_counted_as_unsupported() -> None:
    evidence = [_span("invoice INV-1001 was paid in full on 2026-03-15 via ach")]
    result = ground_answer("The weather in Paris was lovely that day.", evidence)

    assert result.citations == []
    assert len(result.unsupported_assertions) == 1


def test_mixed_answer_reports_an_accurate_unsupported_rate() -> None:
    evidence = [_span("invoice INV-1001 was paid in full on 2026-03-15 via ach")]
    answer = "Invoice INV-1001 was paid in full on 2026-03-15 via ACH. The weather was lovely."
    result = ground_answer(answer, evidence)

    assert len(result.citations) == 1
    assert len(result.unsupported_assertions) == 1
    assert result.unsupported_rate == 0.5


def test_no_evidence_means_every_claim_is_unsupported() -> None:
    result = ground_answer("Invoice INV-1001 was paid in full.", [])
    assert result.citations == []
    assert len(result.unsupported_assertions) == 1


def test_empty_answer_produces_no_claims_at_all() -> None:
    result = ground_answer("", [_span("something")])
    assert result.citations == []
    assert result.unsupported_assertions == []
    assert result.unsupported_rate == 0.0


def test_decimal_point_does_not_split_the_sentence() -> None:
    evidence = [_span("the deductible is $500.00 on this policy")]
    answer = "The deductible is $500.00 on this policy."
    result = ground_answer(answer, evidence)

    assert result.unsupported_assertions == []
    assert len(result.citations) == 1
    assert result.citations[0].assertion == answer


def test_name_initial_does_not_split_the_sentence() -> None:
    evidence = [_span("the policyholder is J. Rivera on this policy")]
    answer = "The policyholder is J. Rivera on this policy."
    result = ground_answer(answer, evidence)

    assert result.unsupported_assertions == []
    assert len(result.citations) == 1
    assert result.citations[0].assertion == answer


def test_negation_is_not_cited_as_support() -> None:
    """A sentence that borrows the source's words while inverting its
    meaning must not receive a citation. Token overlap alone cannot see
    this, which is why grounding checks negation parity separately.
    """
    evidence = [_span("payment PAY-1 received 2026-03-15 for $1,240.00 via ach, paid in full")]

    result = ground_answer(
        "Payment PAY-1 was not received via ACH, and is not paid in full.", evidence
    )

    assert result.citations == []
    assert len(result.unsupported_assertions) == 1


def test_a_substituted_value_is_not_cited_as_support() -> None:
    evidence = [_span("the deductible on this policy is $500.00 for the current term")]

    result = ground_answer(
        "The deductible on this policy is $9,999.00 for the current term.", evidence
    )

    assert result.citations == []


def test_a_faithful_restatement_is_still_cited() -> None:
    """The contradiction guards must not reject honest paraphrase, or
    they would trade false citations for false refusals.
    """
    evidence = [_span("the deductible on this policy is $500.00 for the current term")]

    result = ground_answer(
        "The deductible on this policy is $500.00 for the current term.", evidence
    )

    assert len(result.citations) == 1
    assert result.unsupported_assertions == []


def test_currency_formatting_does_not_break_the_value_check() -> None:
    """$1,240.00 and 1240 are the same value, so a thousands separator or
    a trailing zero must not read as a substituted number.
    """
    evidence = [_span("invoice INV-1001 was issued for $1,240.00 on 2026-03-14")]

    result = ground_answer("Invoice INV-1001 was issued for 1240 on 2026-03-14.", evidence)

    assert len(result.citations) == 1
