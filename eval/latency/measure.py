"""Measures p50 and p99 latency per model against the tenant's declared
latency budget.

Requires LLM_API_KEY and makes real network calls: never run by tests
or CI, which only ever read the committed artifact this writes.
"""

import json
import time
from pathlib import Path
from typing import Any

from outpost.llm.base import Message
from outpost.llm.errors import ProviderError
from outpost.llm.openai_compatible import OpenAICompatibleProvider

ARTIFACT_PATH = Path(__file__).resolve().parents[2] / "eval" / "artifacts" / "latency_results.json"
MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
SAMPLE_COUNT = 8
BUDGET_MS = 8000
REQUEST_TIMEOUT_SECONDS = 30.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))
    return ordered[index]


def measure_model(model: str, *, sample_count: int = SAMPLE_COUNT) -> dict[str, Any]:
    # A request that errors or times out is recorded as a failure rather
    # than aborting the whole measurement: a model that only sometimes
    # answers is itself a real, worth-reporting result, not just noise
    # to retry past.
    provider = OpenAICompatibleProvider(model=model, timeout_seconds=REQUEST_TIMEOUT_SECONDS)
    samples_ms: list[float] = []
    failures = 0
    for i in range(sample_count):
        start = time.monotonic()
        try:
            provider.complete(
                [Message(role="user", content=f"Reply with the single word ok. ({i})")]
            )
            samples_ms.append((time.monotonic() - start) * 1000)
        except ProviderError:
            failures += 1

    p50 = _percentile(samples_ms, 50)
    p99 = _percentile(samples_ms, 99)
    return {
        "sample_count": sample_count,
        "successful_samples": len(samples_ms),
        "failed_samples": failures,
        "samples_ms": [round(sample, 1) for sample in samples_ms],
        "p50_ms": round(p50, 1),
        "p99_ms": round(p99, 1),
        "budget_ms": BUDGET_MS,
        "within_budget": bool(samples_ms) and p99 <= BUDGET_MS,
    }


def main() -> None:
    results = {}
    for model in MODELS:
        stats = measure_model(model)
        results[model] = stats
        print(
            f"{model}: p50={stats['p50_ms']}ms p99={stats['p99_ms']}ms "
            f"failures={stats['failed_samples']}/{stats['sample_count']}"
        )

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
