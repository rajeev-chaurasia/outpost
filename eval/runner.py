"""Regenerates evaluation artifacts, and verifies committed ones.

Two kinds of artifact live in eval/artifacts/:

Deterministic ones (isolation, grounding, degradation) replay committed
fixtures and can be regenerated with no api key and no network, so
--verify-artifacts recomputes them and fails if the committed file
disagrees. That is the gate that keeps the readme's numbers honest.

Live ones (latency, onboarding) cannot be recomputed on demand: latency
measures real network calls, and onboarding includes a wall-clock time
and an intervention count only the person who ran it can supply. Those
are verified by hash against manifest.json, which at least catches an
artifact edited by hand after the fact.
"""

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from eval.degradation.force import run_all as run_degradation
from eval.grounding.entailment import load_cases as load_entailment_cases
from eval.grounding.entailment import run as run_entailment
from eval.grounding.score import score as run_grounding
from eval.isolation.adversarial import build_multi_tenant_index, load_cases, run_isolation_suite
from eval.isolation.adversarial import summarize as summarize_isolation

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "eval" / "artifacts"
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"

DETERMINISTIC_ARTIFACTS = (
    "isolation_results",
    "grounding_results",
    "degradation_results",
    "entailment_results",
)
LIVE_ARTIFACTS = ("latency_results", "onboarding_results")


def _artifact_path(name: str) -> Path:
    return ARTIFACTS_DIR / f"{name}.json"


def _dump(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regenerate_isolation() -> Any:
    lexical_index, dense_store = build_multi_tenant_index()
    return summarize_isolation(run_isolation_suite(lexical_index, dense_store, load_cases()))


def regenerate_grounding() -> Any:
    with tempfile.TemporaryDirectory() as tmp_dir:
        return run_grounding(Path(tmp_dir) / "audit.sqlite")


def regenerate_degradation() -> Any:
    with tempfile.TemporaryDirectory() as tmp_dir:
        results = run_degradation(Path(tmp_dir) / "audit.sqlite")
    correct = sum(1 for r in results.values() if r["correct"])
    return {"correct_rung_rate": correct / len(results), "scenarios": results}


def regenerate_entailment() -> Any:
    return run_entailment(load_entailment_cases())


REGENERATORS = {
    "isolation_results": regenerate_isolation,
    "grounding_results": regenerate_grounding,
    "degradation_results": regenerate_degradation,
    "entailment_results": regenerate_entailment,
}


def write_manifest() -> dict[str, str]:
    manifest = {
        f"{name}.json": _sha256(_artifact_path(name))
        for name in (*DETERMINISTIC_ARTIFACTS, *LIVE_ARTIFACTS)
        if _artifact_path(name).exists()
    }
    MANIFEST_PATH.write_text(_dump(manifest))
    return manifest


def regenerate_all() -> None:
    for name, regenerate in REGENERATORS.items():
        _artifact_path(name).write_text(_dump(regenerate()))
        print(f"regenerated {name}.json")
    write_manifest()
    print(f"wrote manifest with {len(json.loads(MANIFEST_PATH.read_text()))} entries")


def verify_artifacts() -> int:
    failures: list[str] = []

    for name, regenerate in REGENERATORS.items():
        path = _artifact_path(name)
        if not path.exists():
            failures.append(f"{name}.json is missing")
            continue
        if _dump(regenerate()) != path.read_text():
            failures.append(f"{name}.json does not match a fresh run")

    if not MANIFEST_PATH.exists():
        failures.append("manifest.json is missing")
    else:
        manifest = json.loads(MANIFEST_PATH.read_text())
        for filename, expected_hash in manifest.items():
            path = ARTIFACTS_DIR / filename
            if not path.exists():
                failures.append(f"{filename} is in the manifest but missing from disk")
            elif _sha256(path) != expected_hash:
                failures.append(f"{filename} does not match its manifest hash")
        for name in LIVE_ARTIFACTS:
            if f"{name}.json" not in manifest:
                failures.append(f"{name}.json is missing from the manifest")

    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        return 1
    print("all artifacts verified")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-artifacts",
        action="store_true",
        help="recompute deterministic artifacts and check every artifact's manifest hash",
    )
    parser.add_argument(
        "--all", action="store_true", help="regenerate deterministic artifacts and the manifest"
    )
    args = parser.parse_args()

    if args.verify_artifacts:
        return verify_artifacts()
    if args.all:
        regenerate_all()
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
