"""Tenant configuration model and the yaml loader that produces it."""

from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field
from yaml.nodes import MappingNode

from outpost.ontology.schema import TenantOntology
from outpost.ontology.validate import validate_raw_config

_LINE_KEY = "__line__"


class SourceConfig(BaseModel):
    """Where one slice of a tenant's data comes from and how to interpret it."""

    model_config = ConfigDict(extra="forbid")

    id: str
    connector: Literal["csv_export", "pdf_text", "rest_mock"]
    path: str
    entity: str | None = None
    field_map: dict[str, list[str]] = Field(default_factory=dict)


class ActionsConfig(BaseModel):
    """Which agent actions a tenant permits, and which need human review."""

    model_config = ConfigDict(extra="forbid")

    allowed: list[str]
    requires_review: list[str] = Field(default_factory=list)


class BudgetConfig(BaseModel):
    """The latency and token ceilings a tenant's requests must stay inside."""

    model_config = ConfigDict(extra="forbid")

    latency_p99_ms: int
    max_tokens_per_request: int


class TenantConfig(BaseModel):
    """Everything outpost needs to know to serve one tenant."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    display_name: str
    ontology: TenantOntology
    sources: list[SourceConfig]
    terminology: dict[str, str] = Field(default_factory=dict)
    actions: ActionsConfig
    budget: BudgetConfig


class _LineTrackingLoader(yaml.SafeLoader):
    """A yaml loader that stamps every mapping with the line it started on.

    Cross-reference validation needs to tell a tenant admin which line to
    fix. Plain pydantic validation errors do not carry yaml positions, so
    the line number is captured here, used in validate.py, then stripped
    before the dict is handed to pydantic.
    """


def _construct_mapping(loader: yaml.SafeLoader, node: MappingNode) -> dict[str, Any]:
    # Tenant config yaml only ever uses string keys, so the cast just narrows
    # the stub's overly general dict[Hashable, Any] back to what it actually is.
    mapping = cast("dict[str, Any]", loader.construct_mapping(node))
    mapping[_LINE_KEY] = node.start_mark.line + 1
    return mapping


_LineTrackingLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def _strip_line_markers(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_line_markers(v) for k, v in value.items() if k != _LINE_KEY}
    if isinstance(value, list):
        return [_strip_line_markers(item) for item in value]
    return value


def load_tenant_config(path: Path) -> TenantConfig:
    """Load and validate a tenant config.

    Raises ConfigValidationError, naming the offending key and yaml line,
    if the file is well-formed yaml but a cross-reference does not hold
    (a relation or source pointing at an entity that was never declared).
    """
    raw = yaml.load(path.read_text(), Loader=_LineTrackingLoader)
    validate_raw_config(raw)
    return TenantConfig.model_validate(_strip_line_markers(raw))
