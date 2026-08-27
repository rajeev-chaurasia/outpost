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
    return set(_WORD_RE.findall(text.lower()))


def _overlap_ratio(assertion_tokens: set[str], span_tokens: set[str]) -> float:
    if not assertion_tokens:
        return 0.0
    return len(assertion_tokens & span_tokens) / len(assertion_tokens)


def ground_answer(
    answer: str, evidence_spans: list[Span], *, overlap_threshold: float = 0.6
) -> GroundingResult:
    """Splits answer into sentences and binds each one to whichever
    evidence span best supports it, if any span clears the threshold.
    """
    span_tokens = [(span, _tokens(span.text)) for span in evidence_spans]
    citations: list[Citation] = []
    unsupported: list[str] = []

    for raw_sentence in _SENTENCE_RE.findall(answer):
        sentence = raw_sentence.strip()
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
