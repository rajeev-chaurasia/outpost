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
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

# Cues that flip a sentence's polarity relative to its source.
_NEGATIONS = frozenset(
    {
        "no",
        "not",
        "never",
        "cannot",
        "cant",
        "wasnt",
        "isnt",
        "arent",
        "didnt",
        "doesnt",
        "without",
        "denied",
        "refused",
        "declined",
        "unpaid",
        "outstanding",
    }
)
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


def _numbers(text: str) -> set[str]:
    """Numeric values in text, with thousands separators removed and
    trailing zeros normalized, so $1,240.00 and 1240 compare equal.
    """
    values = set()
    for raw in _NUMBER_RE.findall(text.replace(",", "")):
        normalized = raw.rstrip(".")
        if "." in normalized:
            normalized = normalized.rstrip("0").rstrip(".")
        values.add(normalized or "0")
    return values


def _negation_cues(text: str) -> set[str]:
    return {word for word in _WORD_RE.findall(text.lower()) if word in _NEGATIONS}


def _contradicts(sentence: str, span_text: str) -> bool:
    """Rejects a span that shares the sentence's vocabulary but not its
    meaning.

    Token overlap cannot see either of these, and both flip meaning while
    leaving overlap almost unchanged:

    introduced negation
        "was not paid in full" against a source saying it was paid.
    substituted value
        "$9,999.00" against a source saying $1,240.00.

    Neither check is entailment. They catch the two ways a borrowed
    sentence most often inverts its source, and anything subtler still
    gets through, which the entailment eval measures.
    """
    if _negation_cues(sentence) - _negation_cues(span_text):
        return True
    return bool(_numbers(sentence) - _numbers(span_text))


def ground_answer(
    answer: str, evidence_spans: list[Span], *, overlap_threshold: float = 0.6
) -> GroundingResult:
    """Splits answer into sentences and binds each one to whichever
    evidence span best supports it, if any span clears the threshold and
    does not contradict the sentence.
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
            if _contradicts(sentence, span.text):
                continue
            ratio = _overlap_ratio(assertion_tokens, tokens)
            if ratio > best_ratio:
                best_ratio = ratio
                best_span = span

        if best_span is not None and best_ratio >= overlap_threshold:
            citations.append(Citation(assertion=sentence, span=best_span))
        else:
            unsupported.append(sentence)

    return GroundingResult(citations=citations, unsupported_assertions=unsupported)
