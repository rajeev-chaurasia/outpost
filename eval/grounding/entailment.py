"""Measures whether a citation means the source supports the sentence, or
only that the sentence reuses the source's words.

ground_answer binds a sentence to a span on token overlap. Overlap cannot
see negation or a substituted value, so a sentence that borrows the
source's vocabulary while contradicting it scores about as well as one
that restates it. This puts a number on that gap instead of asserting it
is small.

Runs against ground_answer directly with hand-written spans, so it needs
no model, no embeddings, and no network.
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from outpost.agent.ground import ground_answer
from outpost.retrieval.document import Span

CASES_PATH = Path(__file__).resolve().parent / "entailment.yaml"
ARTIFACT_PATH = (
    Path(__file__).resolve().parents[2] / "eval" / "artifacts" / "entailment_results.json"
)


@dataclass(frozen=True)
class EntailmentCase:
    tenant_id: str
    category: str
    source: str
    assertion: str
    should_cite: bool


def load_cases() -> list[EntailmentCase]:
    raw = yaml.safe_load(CASES_PATH.read_text())
    return [EntailmentCase(**case) for case in raw["cases"]]


def _cites(case: EntailmentCase) -> bool:
    span = Span(
        source_id="entailment",
        document_id=f"{case.tenant_id}:entailment",
        start=0,
        end=len(case.source),
        text=case.source,
    )
    return bool(ground_answer(case.assertion, [span]).citations)


def run(cases: list[EntailmentCase]) -> dict[str, Any]:
    per_category: dict[str, Counter[str]] = {}
    failures: list[dict[str, str]] = []

    for case in cases:
        cited = _cites(case)
        bucket = per_category.setdefault(case.category, Counter())
        bucket["cases"] += 1
        bucket["cited"] += int(cited)
        if cited == case.should_cite:
            bucket["correct"] += 1
        else:
            failures.append(
                {
                    "tenant_id": case.tenant_id,
                    "category": case.category,
                    "assertion": case.assertion,
                    "expected_cite": str(case.should_cite),
                    "actual_cite": str(cited),
                }
            )

    # A false citation is the dangerous direction: the answer is presented
    # as sourced when the source does not support it.
    adversarial = [c for c in cases if not c.should_cite and c.category != "unrelated"]
    false_citations = sum(1 for c in adversarial if _cites(c))

    return {
        "case_count": len(cases),
        "correct": sum(b["correct"] for b in per_category.values()),
        "per_category": {
            name: {
                "cases": b["cases"],
                "cited": b["cited"],
                "correct": b["correct"],
            }
            for name, b in sorted(per_category.items())
        },
        "adversarial_case_count": len(adversarial),
        "false_citations_on_adversarial": false_citations,
        "false_citation_rate": (
            round(false_citations / len(adversarial), 3) if adversarial else 0.0
        ),
        "misclassified": failures,
    }


def main() -> None:
    results = run(load_cases())
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
