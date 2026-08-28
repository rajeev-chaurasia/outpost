"""Measures per-model latency against the declared budget, across three
arms, to establish what actually bounds what a user waits.

raw
    Direct calls with a generous transport timeout. Shows the endpoint's
    natural distribution, tail included.
paced
    A minimum gap between calls, testing whether the tail is caused by
    issuing calls back to back. Measures the provider's own response
    time, not the pacer's wait, so this is about the endpoint rather
    than about moving the wait somewhere else.
budget_enforced
    What the system actually runs: the transport deadline set to the
    tenant's budget, falling back to the secondary model when the
    primary is cut off. Measures user-visible time end to end, fallback
    included, which is the number that matters.

Percentiles are reported as p50 and p90 with the max stated separately.
SAMPLE_COUNT samples cannot estimate a p99, and reporting max-of-n under
that name would overstate what was measured.

Requires LLM_API_KEY and makes real network calls: never run by tests
or CI, which only ever read the committed artifact this writes.
"""

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from outpost.llm.base import Completion, Message, Provider, ToolSpec
from outpost.llm.budget import build_budgeted_provider
from outpost.llm.errors import ProviderError
from outpost.llm.fallback import FallbackProvider
from outpost.llm.openai_compatible import OpenAICompatibleProvider
from outpost.llm.pacing import PacedProvider

ARTIFACT_PATH = Path(__file__).resolve().parents[2] / "eval" / "artifacts" / "latency_results.json"
PRIMARY_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"
MODELS = [PRIMARY_MODEL, FALLBACK_MODEL]
SAMPLE_COUNT = 20
BUDGET_MS = 8000
RAW_TIMEOUT_SECONDS = 30.0
PACING_INTERVAL_SECONDS = 2.0
MAX_TOKENS = 4000


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))
    return ordered[index]


def _stats(samples_ms: list[float], failures: Counter[str], sample_count: int) -> dict[str, Any]:
    return {
        "sample_count": sample_count,
        "successful_samples": len(samples_ms),
        "failed_samples": sum(failures.values()),
        "failure_kinds": dict(failures),
        "p50_ms": round(_percentile(samples_ms, 50), 1),
        "p90_ms": round(_percentile(samples_ms, 90), 1),
        "max_ms": round(max(samples_ms), 1) if samples_ms else 0.0,
        "samples_ms": [round(ms, 1) for ms in samples_ms],
    }


def _prompt(i: int) -> list[Message]:
    return [Message(role="user", content=f"Reply with the single word ok. ({i})")]


def _classify(exc: ProviderError) -> str:
    """Names the failure mode so the artifact records what happened
    rather than leaving the mechanism to be guessed at afterward.
    """
    detail = exc.detail.lower()
    if "timed out" in detail or "timeout" in detail:
        return "client_timeout"
    for status in ("429", "500", "502", "503", "504"):
        if status in detail:
            return f"http_{status}"
    return "other"


def _run_arm(provider: Provider, sample_count: int) -> tuple[list[float], Counter[str]]:
    # A request that errors or times out is recorded as a failure rather
    # than aborting the whole measurement: a model that only sometimes
    # answers is itself a real, worth-reporting result.
    samples_ms: list[float] = []
    failures: Counter[str] = Counter()
    for i in range(sample_count):
        start = time.monotonic()
        try:
            provider.complete(_prompt(i))
            samples_ms.append((time.monotonic() - start) * 1000)
        except ProviderError as exc:
            failures[_classify(exc)] += 1
    return samples_ms, failures


class _TimedInner:
    """Times only the wrapped provider's own call, so the paced arm's
    numbers exclude the interval the pacer waited.
    """

    def __init__(self, inner: Provider) -> None:
        self.inner = inner
        self.service_ms: list[float] = []

    def complete(
        self, messages: list[Message], *, tools: list[ToolSpec] | None = None
    ) -> Completion:
        start = time.monotonic()
        completion = self.inner.complete(messages, tools=tools)
        # Recorded only on success. Timing a call that timed out would
        # fold the client deadline into the latency distribution and make
        # a failing arm look merely slow.
        self.service_ms.append((time.monotonic() - start) * 1000)
        return completion


def measure_raw(model: str, sample_count: int) -> dict[str, Any]:
    provider = OpenAICompatibleProvider(model=model, timeout_seconds=RAW_TIMEOUT_SECONDS)
    samples, failures = _run_arm(provider, sample_count)
    stats = _stats(samples, failures, sample_count)
    stats["within_budget"] = bool(samples) and stats["max_ms"] <= BUDGET_MS and not failures
    return stats


def measure_paced(model: str, sample_count: int) -> dict[str, Any]:
    timed = _TimedInner(OpenAICompatibleProvider(model=model, timeout_seconds=RAW_TIMEOUT_SECONDS))
    _, failures = _run_arm(
        PacedProvider(inner=timed, min_interval_seconds=PACING_INTERVAL_SECONDS), sample_count
    )
    stats = _stats(timed.service_ms, failures, sample_count)
    stats["pacing_interval_seconds"] = PACING_INTERVAL_SECONDS
    stats["within_budget"] = (
        bool(timed.service_ms) and stats["max_ms"] <= BUDGET_MS and not failures
    )
    return stats


def measure_budget_enforced(sample_count: int) -> dict[str, Any]:
    """The composed path the served app runs: a primary whose transport
    deadline is the budget, falling back to the secondary when cut off.
    """
    fallbacks = 0
    samples_ms: list[float] = []
    failures: Counter[str] = Counter()

    for i in range(sample_count):
        provider = FallbackProvider(
            primary=build_budgeted_provider(
                PRIMARY_MODEL, latency_p99_ms=BUDGET_MS, max_tokens_per_request=MAX_TOKENS
            ),
            secondary=build_budgeted_provider(
                FALLBACK_MODEL, latency_p99_ms=BUDGET_MS, max_tokens_per_request=MAX_TOKENS
            ),
        )
        start = time.monotonic()
        try:
            provider.complete(_prompt(i))
            samples_ms.append((time.monotonic() - start) * 1000)
        except ProviderError as exc:
            failures[_classify(exc)] += 1
        if provider.fell_back:
            fallbacks += 1

    stats = _stats(samples_ms, failures, sample_count)
    stats["fallbacks"] = fallbacks
    stats["answered"] = len(samples_ms)
    # The user-visible worst case is the primary being cut off at the
    # budget plus the secondary answering, so the ceiling checked here is
    # two budgets, not one.
    stats["within_two_budgets"] = bool(samples_ms) and stats["max_ms"] <= 2 * BUDGET_MS
    return stats


def main() -> None:
    results: dict[str, Any] = {
        "budget_ms": BUDGET_MS,
        "sample_count": SAMPLE_COUNT,
        "percentile_note": (
            f"{SAMPLE_COUNT} samples cannot estimate a p99; p50 and p90 are reported "
            "with the max stated separately as the tail."
        ),
        "per_model": {},
    }

    for model in MODELS:
        raw = measure_raw(model, SAMPLE_COUNT)
        paced = measure_paced(model, SAMPLE_COUNT)
        results["per_model"][model] = {"raw": raw, "paced": paced}
        print(
            f"{model}\n"
            f"  raw:   p50={raw['p50_ms']}ms p90={raw['p90_ms']}ms max={raw['max_ms']}ms "
            f"failures={raw['failed_samples']}/{raw['sample_count']}\n"
            f"  paced: p50={paced['p50_ms']}ms p90={paced['p90_ms']}ms max={paced['max_ms']}ms "
            f"failures={paced['failed_samples']}/{paced['sample_count']}"
        )

    enforced = measure_budget_enforced(SAMPLE_COUNT)
    results["budget_enforced"] = enforced
    print(
        f"budget_enforced (primary cut off at budget, fallback to secondary)\n"
        f"  p50={enforced['p50_ms']}ms p90={enforced['p90_ms']}ms max={enforced['max_ms']}ms "
        f"answered={enforced['answered']}/{enforced['sample_count']} "
        f"fallbacks={enforced['fallbacks']}"
    )

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
