"""Phase 1 done-tests: both tenant configs load, cross-reference validation
fires with a line number on a bad reference, and no tenant vocabulary
leaks into the core package.
"""

import re
from pathlib import Path

import pytest

from outpost.ontology import ConfigValidationError, load_tenant_config

TENANTS_DIR = Path(__file__).resolve().parents[1] / "tenants"


def test_dealer_ar_and_claims_intake_load_with_different_entities() -> None:
    dealer = load_tenant_config(TENANTS_DIR / "dealer_ar" / "config.yaml")
    claims = load_tenant_config(TENANTS_DIR / "claims_intake" / "config.yaml")

    dealer_entities = {entity.name for entity in dealer.ontology.entities}
    claims_entities = {entity.name for entity in claims.ontology.entities}

    assert dealer_entities == {"invoice", "payment", "account"}
    assert claims_entities == {"claim", "policy", "adjuster_note"}
    assert dealer_entities.isdisjoint(claims_entities)
    assert dealer.terminology != claims.terminology


def test_source_with_unknown_entity_type_fails_with_line_number(tmp_path: Path) -> None:
    bad_config = tmp_path / "config.yaml"
    bad_config.write_text(
        """
tenant_id: broken
display_name: Broken Tenant
ontology:
  entities:
    - name: widget
      key: widget_id
      fields: [widget_id]
  relations: []
sources:
  - id: widgets
    connector: csv_export
    path: fixtures/widgets.csv
    entity: gadget
terminology: {}
actions:
  allowed: []
budget:
  latency_p99_ms: 8000
  max_tokens_per_request: 4000
"""
    )

    with pytest.raises(ConfigValidationError) as exc_info:
        load_tenant_config(bad_config)

    assert "gadget" in str(exc_info.value)
    assert exc_info.value.line is not None


def test_relation_pointing_at_nonexistent_entity_fails(tmp_path: Path) -> None:
    bad_config = tmp_path / "config.yaml"
    bad_config.write_text(
        """
tenant_id: broken
display_name: Broken Tenant
ontology:
  entities:
    - name: widget
      key: widget_id
      fields: [widget_id]
  relations:
    - name: powers
      from: widget
      to: gizmo
sources: []
terminology: {}
actions:
  allowed: []
budget:
  latency_p99_ms: 8000
  max_tokens_per_request: 4000
"""
    )

    with pytest.raises(ConfigValidationError) as exc_info:
        load_tenant_config(bad_config)

    assert "gizmo" in str(exc_info.value)
    assert exc_info.value.line is not None


BANNED_WORDS = [
    "invoice",
    "payment",
    "claim",
    "policy",
    "adjuster",
    "dealer",
    "premium",
    "deductible",
]
_LEAK_PATTERN = re.compile(r"\b(" + "|".join(BANNED_WORDS) + r")\b", re.IGNORECASE)


def test_no_tenant_vocabulary_leaks_into_core() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "outpost"
    violations = [
        f"{path}: {match.group()!r}"
        for path in src_root.rglob("*.py")
        for match in _LEAK_PATTERN.finditer(path.read_text())
    ]
    assert not violations, "tenant vocabulary leaked into core:\n" + "\n".join(violations)
