"""Artifact gate tests: every published number has a committed artifact,
every artifact is in the manifest, and the deterministic ones still match
a fresh run.

This is what keeps the readme honest, so it is checked here as well as in
CI rather than trusted.
"""

import json
from pathlib import Path

from eval.runner import (
    ARTIFACTS_DIR,
    DETERMINISTIC_ARTIFACTS,
    LIVE_ARTIFACTS,
    MANIFEST_PATH,
    REGENERATORS,
    _dump,
    _sha256,
    verify_artifacts,
)


def test_every_expected_artifact_exists() -> None:
    for name in (*DETERMINISTIC_ARTIFACTS, *LIVE_ARTIFACTS):
        assert (ARTIFACTS_DIR / f"{name}.json").exists(), name


def test_manifest_covers_every_artifact() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    expected = {f"{name}.json" for name in (*DETERMINISTIC_ARTIFACTS, *LIVE_ARTIFACTS)}
    assert set(manifest) == expected


def test_manifest_hashes_match_the_files_on_disk() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    for filename, expected_hash in manifest.items():
        assert _sha256(ARTIFACTS_DIR / filename) == expected_hash, filename


def test_deterministic_artifacts_match_a_fresh_run() -> None:
    for name, regenerate in REGENERATORS.items():
        committed = (ARTIFACTS_DIR / f"{name}.json").read_text()
        assert _dump(regenerate()) == committed, name


def test_verify_artifacts_passes_on_the_committed_state() -> None:
    assert verify_artifacts() == 0


def test_isolation_artifact_reports_zero_leaks() -> None:
    payload = json.loads((ARTIFACTS_DIR / "isolation_results.json").read_text())
    assert payload["total_leaks"] == 0
    assert payload["zero_leak_invariant_met"] is True
    # The measured argument for traversal-time filtering: post-filtering
    # returns strictly fewer authorized results on the same queries.
    assert payload["post_filter_authorized_results"] < payload["traversal_authorized_results"]


def test_grounding_artifact_reports_every_tenant_separately() -> None:
    payload = json.loads((ARTIFACTS_DIR / "grounding_results.json").read_text())
    tenants_dir = Path(__file__).resolve().parents[1] / "tenants"
    from outpost.ontology import discover_tenant_ids

    assert set(payload["per_tenant"]) == set(discover_tenant_ids(tenants_dir))


def test_grounding_artifact_scores_enough_assertions_to_mean_something() -> None:
    """A rate over one assertion per tenant is an anecdote. This pins the
    denominator so the unsupported rate cannot quietly shrink back to it.
    """
    payload = json.loads((ARTIFACTS_DIR / "grounding_results.json").read_text())

    assert payload["totals"]["assertions_scored"] >= 12
    for tenant, stats in payload["per_tenant"].items():
        assert stats["assertions_scored"] >= 3, tenant


def test_grounding_artifact_neither_over_refuses_nor_over_answers() -> None:
    """Both rates together, because either alone is gameable: refusing
    everything scores a perfect refusal rate, and answering only easy
    questions scores a perfect unsupported rate.
    """
    payload = json.loads((ARTIFACTS_DIR / "grounding_results.json").read_text())

    assert payload["totals"]["correct_refusal_rate"] == 1.0
    for tenant, stats in payload["per_tenant"].items():
        assert stats["refused_when_it_should_have_answered"] == [], tenant
        assert stats["answered_when_it_should_have_refused"] == [], tenant


def test_entailment_artifact_reports_no_false_citations() -> None:
    """Grounding must not cite a span that contradicts the sentence. The
    adversarial cases are negation-inverted and value-substituted claims
    that share nearly all their vocabulary with the source, so overlap
    alone would cite them.
    """
    payload = json.loads((ARTIFACTS_DIR / "entailment_results.json").read_text())

    assert payload["case_count"] == payload["correct"]
    assert payload["false_citations_on_adversarial"] == 0
    assert payload["false_citation_rate"] == 0.0
    assert payload["misclassified"] == []


def test_entailment_artifact_still_cites_faithful_restatements() -> None:
    """Rejecting every adversarial case would be trivial to achieve by
    citing nothing, so the faithful cases have to still be cited.
    """
    payload = json.loads((ARTIFACTS_DIR / "entailment_results.json").read_text())
    faithful = payload["per_category"]["faithful"]

    assert faithful["cases"] > 0
    assert faithful["cited"] == faithful["cases"]
