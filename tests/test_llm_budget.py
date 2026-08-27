"""Budget done-tests: a completion slower than the tenant's latency
ceiling is rejected rather than accepted, one that uses more tokens
than the ceiling is rejected too, and wrapping a budgeted provider in
FallbackProvider turns a budget breach into an actual fallback, the
same as any other provider failure.
"""

import json
import time
from dataclasses import dataclass

import pytest

from outpost.llm.base import Completion, Message, ToolSpec, Usage
from outpost.llm.budget import BudgetedProvider
from outpost.llm.errors import LatencyBudgetExceededError, TokenBudgetExceededError
from outpost.llm.fallback import FallbackProvider


@dataclass
class _SlowProvider:
    delay_seconds: float
    completion: Completion

    def complete(
        self, messages: list[Message], *, tools: list[ToolSpec] | None = None
    ) -> Completion:
        time.sleep(self.delay_seconds)
        return self.completion


@dataclass
class _FixedProvider:
    completion: Completion

    def complete(
        self, messages: list[Message], *, tools: list[ToolSpec] | None = None
    ) -> Completion:
        return self.completion


def _completion(prompt_tokens: int = 10, completion_tokens: int = 10) -> Completion:
    return Completion(
        content="ok",
        tool_calls=[],
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        model="m",
    )


def test_completion_within_budget_passes_through() -> None:
    provider = BudgetedProvider(
        inner=_FixedProvider(_completion()),
        model="m",
        latency_p99_ms=8000,
        max_tokens_per_request=4000,
    )
    completion = provider.complete([Message(role="user", content="hi")])
    assert completion.content == "ok"


def test_completion_over_latency_budget_raises() -> None:
    provider = BudgetedProvider(
        inner=_SlowProvider(delay_seconds=0.05, completion=_completion()),
        model="m",
        latency_p99_ms=10,
        max_tokens_per_request=4000,
    )
    with pytest.raises(LatencyBudgetExceededError) as exc_info:
        provider.complete([Message(role="user", content="hi")])
    assert exc_info.value.budget_ms == 10


def test_completion_over_token_budget_raises() -> None:
    provider = BudgetedProvider(
        inner=_FixedProvider(_completion(prompt_tokens=3000, completion_tokens=3000)),
        model="m",
        latency_p99_ms=8000,
        max_tokens_per_request=4000,
    )
    with pytest.raises(TokenBudgetExceededError) as exc_info:
        provider.complete([Message(role="user", content="hi")])
    assert exc_info.value.total_tokens == 6000


def test_latency_budget_breach_triggers_fallback() -> None:
    primary = BudgetedProvider(
        inner=_SlowProvider(delay_seconds=0.05, completion=_completion()),
        model="primary",
        latency_p99_ms=10,
        max_tokens_per_request=4000,
    )
    secondary = _FixedProvider(_completion())
    fallback = FallbackProvider(primary=primary, secondary=secondary)

    completion = fallback.complete([Message(role="user", content="hi")])

    assert fallback.fell_back is True
    assert completion.content == "ok"


def test_latency_artifact_has_expected_shape() -> None:
    """The committed latency_results.json is generated live and its
    values will change on every remeasurement, so this checks its
    structure, not its numbers: asserting a live-measured value here
    would make CI depend on network conditions at measurement time,
    which is exactly what CI must never do.
    """
    from eval.latency.measure import ARTIFACT_PATH, MODELS

    payload = json.loads(ARTIFACT_PATH.read_text())
    assert {"budget_ms", "sample_count", "percentile_note", "per_model", "budget_enforced"} <= set(
        payload
    )
    assert set(payload["per_model"]) == set(MODELS)

    arm_fields = {
        "sample_count",
        "successful_samples",
        "failed_samples",
        "samples_ms",
        "p50_ms",
        "p90_ms",
        "max_ms",
    }
    for model in MODELS:
        for arm in ("raw", "paced"):
            assert arm_fields <= set(payload["per_model"][model][arm]), (model, arm)

    enforced = payload["budget_enforced"]
    assert arm_fields <= set(enforced)
    assert {"fallbacks", "answered", "within_two_budgets"} <= set(enforced)


def test_latency_artifact_does_not_claim_a_p99_it_cannot_estimate() -> None:
    """A p99 needs far more samples than this harness takes, so the
    artifact must report p50, p90, and max instead. This pins that,
    because reporting max-of-n as p99 is exactly the mistake the
    earlier version of this harness made.
    """
    from eval.latency.measure import ARTIFACT_PATH

    payload = json.loads(ARTIFACT_PATH.read_text())
    assert "cannot estimate a p99" in payload["percentile_note"]
    for arms in payload["per_model"].values():
        for arm in arms.values():
            assert "p99_ms" not in arm


def test_build_budgeted_provider_sets_the_transport_deadline_to_the_budget() -> None:
    """The post-hoc elapsed check cannot bound what a user waits, since
    it fires only after the response has already arrived. The transport
    timeout is the mechanism, so it must actually be set from the budget.
    """
    from outpost.llm.budget import build_budgeted_provider

    provider = build_budgeted_provider(
        "some-model", latency_p99_ms=8000, max_tokens_per_request=4000
    )

    assert provider.latency_p99_ms == 8000
    assert provider.inner.timeout_seconds == 8.0  # type: ignore[attr-defined]


def test_paced_provider_spaces_calls_by_the_configured_interval() -> None:
    from outpost.llm.pacing import PacedProvider

    inner = _FixedProvider(_completion())
    paced = PacedProvider(inner=inner, min_interval_seconds=0.05)

    start = time.monotonic()
    for _ in range(3):
        paced.complete([Message(role="user", content="hi")])
    elapsed = time.monotonic() - start

    # First call is immediate, the next two each wait one interval.
    assert elapsed >= 0.10


def test_paced_provider_does_not_delay_the_first_call() -> None:
    from outpost.llm.pacing import PacedProvider

    paced = PacedProvider(inner=_FixedProvider(_completion()), min_interval_seconds=5.0)

    start = time.monotonic()
    paced.complete([Message(role="user", content="hi")])

    assert time.monotonic() - start < 1.0
