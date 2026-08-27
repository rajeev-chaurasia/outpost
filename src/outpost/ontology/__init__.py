"""Tenant ontology schema and config loading."""

from outpost.ontology.config import (
    ActionsConfig,
    BudgetConfig,
    SourceConfig,
    TenantConfig,
    discover_tenant_ids,
    load_tenant_config,
)
from outpost.ontology.errors import ConfigValidationError, OntologyError
from outpost.ontology.schema import EntityType, RelationType, TenantOntology

__all__ = [
    "ActionsConfig",
    "BudgetConfig",
    "ConfigValidationError",
    "EntityType",
    "OntologyError",
    "RelationType",
    "SourceConfig",
    "TenantConfig",
    "TenantOntology",
    "discover_tenant_ids",
    "load_tenant_config",
]
