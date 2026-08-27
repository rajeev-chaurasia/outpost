"""Cross-reference validation for a raw, still yaml-shaped tenant config.

Runs before the dict is handed to pydantic, while every mapping still
carries the __line__ marker the config loader's yaml loader attaches, so a
bad reference can be reported with the line a tenant admin needs to fix.
"""

from typing import Any

from outpost.ontology.errors import ConfigValidationError


def validate_raw_config(raw: dict[str, Any]) -> None:
    entity_names = {entity["name"] for entity in raw["ontology"]["entities"]}

    for relation in raw["ontology"].get("relations", []):
        for direction in ("from", "to"):
            target = relation.get(direction)
            if target not in entity_names:
                raise ConfigValidationError(
                    f"relation '{relation.get('name')}' references unknown entity "
                    f"'{target}' via '{direction}'",
                    key=direction,
                    line=relation.get("__line__"),
                )

    for source in raw.get("sources", []):
        entity = source.get("entity")
        if entity is not None and entity not in entity_names:
            raise ConfigValidationError(
                f"source '{source.get('id')}' declares unknown entity type '{entity}'",
                key="entity",
                line=source.get("__line__"),
            )
