"""Phase 5 done-tests: each rung can be forced and the correct one
fires, rung 2 names its gaps, rung 3 produces a draft instead of
executing, and rung 5 states what was tried.
"""

from pathlib import Path

from eval.degradation.force import (
    force_action_declined,
    force_full,
    force_partial,
    force_provider_fallback,
    force_refused,
)
from outpost.agent.audit import AuditLog
from outpost.agent.degrade import Rung, describe_refusal, determine_rung
from outpost.agent.plan import Step


def test_determine_rung_priority_order() -> None:
    # provider fallback outranks everything else
    assert (
        determine_rung(
            final_content="x",
            citation_count=1,
            unsupported_count=0,
            declined_actions=["flag_discrepancy"],
            provider_fell_back=True,
        )
        is Rung.PROVIDER_FALLBACK
    )
    # a declined action outranks refusal and partial
    assert (
        determine_rung(
            final_content=None,
            citation_count=0,
            unsupported_count=0,
            declined_actions=["flag_discrepancy"],
            provider_fell_back=False,
        )
        is Rung.ACTION_DECLINED
    )
    # no answer or zero citations means refused
    assert (
        determine_rung(
            final_content=None,
            citation_count=0,
            unsupported_count=0,
            declined_actions=[],
            provider_fell_back=False,
        )
        is Rung.REFUSED
    )
    # some grounded, some not, means partial
    assert (
        determine_rung(
            final_content="x",
            citation_count=1,
            unsupported_count=1,
            declined_actions=[],
            provider_fell_back=False,
        )
        is Rung.PARTIAL
    )
    # fully grounded means full
    assert (
        determine_rung(
            final_content="x",
            citation_count=2,
            unsupported_count=0,
            declined_actions=[],
            provider_fell_back=False,
        )
        is Rung.FULL
    )


def test_describe_refusal_names_the_attempted_steps() -> None:
    steps = [Step(tool_name="search", arguments={"query": "spaceship warranty"}, result=[])]
    message = describe_refusal(steps)
    assert "search" in message
    assert "spaceship warranty" in message


def test_describe_refusal_with_no_steps_says_no_tools_were_tried() -> None:
    message = describe_refusal([])
    assert "no tools were available" in message


def test_force_full_selects_rung_one(tmp_path: Path) -> None:
    result = force_full(AuditLog(tmp_path / "audit.sqlite"))
    assert result.rung is Rung.FULL


def test_force_partial_selects_rung_two_and_names_the_gap(tmp_path: Path) -> None:
    result = force_partial(AuditLog(tmp_path / "audit.sqlite"))
    assert result.rung is Rung.PARTIAL
    assert result.grounding.unsupported_assertions
    assert "warranty" in result.grounding.unsupported_assertions[0]


def test_force_action_declined_selects_rung_three_with_a_draft(tmp_path: Path) -> None:
    result = force_action_declined(AuditLog(tmp_path / "audit.sqlite"))
    assert result.rung is Rung.ACTION_DECLINED
    declined_step = result.plan.steps[0]
    assert declined_step.result["executed"] is False
    assert declined_step.result["draft"] == {"entity_key": "INV-1005", "reason": "short paid"}


def test_force_provider_fallback_selects_rung_four(tmp_path: Path) -> None:
    result = force_provider_fallback(AuditLog(tmp_path / "audit.sqlite"))
    assert result.rung is Rung.PROVIDER_FALLBACK
    assert result.answer == "Handled by the fallback provider."


def test_force_refused_selects_rung_five_and_states_what_was_tried(tmp_path: Path) -> None:
    result = force_refused(AuditLog(tmp_path / "audit.sqlite"))
    assert result.rung is Rung.REFUSED
    assert result.answer is not None
    assert "search" in result.answer
    assert "INV-1001" in result.answer


def test_forced_rungs_are_all_visible_in_the_audit_log(tmp_path: Path) -> None:
    audit_log = AuditLog(tmp_path / "audit.sqlite")
    result = force_action_declined(audit_log)
    stored = audit_log.get(result.request_id)
    assert stored is not None
    assert stored.rung == Rung.ACTION_DECLINED.value
