"""The failure ladder: which rung an answer used, and why.

Rung selection is a strict priority order, not independent flags: a
provider fallback is reported as a fallback even if the answer that
came back from the secondary provider also happens to be fully
grounded, because the fact that the primary provider failed is itself
something worth surfacing.
"""

from enum import IntEnum

from outpost.agent.plan import Step


class Rung(IntEnum):
    """Which of the five failure-ladder behaviors an answer used."""

    FULL = 1
    PARTIAL = 2
    ACTION_DECLINED = 3
    PROVIDER_FALLBACK = 4
    REFUSED = 5


def determine_rung(
    *,
    final_content: str | None,
    citation_count: int,
    unsupported_count: int,
    declined_actions: list[str],
    provider_fell_back: bool,
) -> Rung:
    if provider_fell_back:
        return Rung.PROVIDER_FALLBACK
    if declined_actions:
        return Rung.ACTION_DECLINED
    if final_content is None or citation_count == 0:
        return Rung.REFUSED
    if unsupported_count > 0:
        return Rung.PARTIAL
    return Rung.FULL


def describe_refusal(steps: list[Step]) -> str:
    """What a rung 5 refusal says instead of guessing: exactly what was
    tried, so the ceiling on what the agent could establish is visible,
    not just the fact that it gave up.
    """
    if not steps:
        return (
            "I could not find enough evidence to answer confidently, and no "
            "tools were available to try."
        )
    attempted = "; ".join(f"{step.tool_name}({step.arguments})" for step in steps)
    return f"I could not find enough evidence to answer confidently. I tried: {attempted}."
