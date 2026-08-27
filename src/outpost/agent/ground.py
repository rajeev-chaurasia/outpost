"""Citation binding.

An assertion is one sentence of the final answer. It is grounded if some
retrieved span's text overlaps it above a threshold; that span becomes
the citation. An assertion with no span clearing the threshold is
unsupported, and is counted rather than silently accepted as if it were
grounded.
"""

import re
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from outpost.retrieval.document import Span

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
_WORD_RE = re.compile(r"[a-z0-9]+")
# Two non-terminal uses of "." get mistaken for sentence endings by the
# naive splitter below: a decimal point ($500.00 must not split into
# "$500." and "00") and a name initial (J. Rivera must not split into
# "J." and "Rivera"). Both get protected with a placeholder unlikely to
# appear in real text, then restored after splitting.
_DECIMAL_POINT_RE = re.compile(r"(?<=\d)\.(?=\d)")
_INITIAL_RE = re.compile(r"(?<![A-Za-z])([A-Z])\.(?=\s+[A-Z])")
_PLACEHOLDER = "\x00"

# Filler words that pad the overlap ratio without asserting anything a
# source could support or contradict. Filtered from both sides so a
# clause with a lot of "the"s and "is"es is not scored on those matching
# trivially, and so it is not diluted by them not matching either.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "for",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)


class Citation(BaseModel):
    """One assertion in an answer, bound to the exact source text it came from."""

    model_config = ConfigDict(extra="forbid")

    assertion: str
    span: Span


@dataclass(frozen=True)
class GroundingResult:
    citations: list[Citation]
    unsupported_assertions: list[str]

    @property
    def unsupported_rate(self) -> float:
        total = len(self.citations) + len(self.unsupported_assertions)
        return len(self.unsupported_assertions) / total if total else 0.0


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower())) - _STOPWORDS


def _overlap_ratio(assertion_tokens: set[str], span_tokens: set[str]) -> float:
    if not assertion_tokens:
        return 0.0
    return len(assertion_tokens & span_tokens) / len(assertion_tokens)


def _split_sentences(answer: str) -> list[str]:
    protected = _DECIMAL_POINT_RE.sub(_PLACEHOLDER, answer)
    protected = _INITIAL_RE.sub(lambda match: match.group(1) + _PLACEHOLDER, protected)
    return [
        raw_sentence.replace(_PLACEHOLDER, ".").strip()
        for raw_sentence in _SENTENCE_RE.findall(protected)
    ]


def ground_answer(
    answer: str, evidence_spans: list[Span], *, overlap_threshold: float = 0.6
) -> GroundingResult:
    """Splits answer into sentences and binds each one to whichever
    evidence span best supports it, if any span clears the threshold.
    """
    span_tokens = [(span, _tokens(span.text)) for span in evidence_spans]
    citations: list[Citation] = []
    unsupported: list[str] = []

    for sentence in _split_sentences(answer):
        if not sentence:
            continue

        assertion_tokens = _tokens(sentence)
        best_span: Span | None = None
        best_ratio = 0.0
        for span, tokens in span_tokens:
            ratio = _overlap_ratio(assertion_tokens, tokens)
            if ratio > best_ratio:
                best_ratio = ratio
                best_span = span

        if best_span is not None and best_ratio >= overlap_threshold:
            citations.append(Citation(assertion=sentence, span=best_span))
        else:
            unsupported.append(sentence)

    return GroundingResult(citations=citations, unsupported_assertions=unsupported)
