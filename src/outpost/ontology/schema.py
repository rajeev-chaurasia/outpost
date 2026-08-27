"""The ontology sub-schema: the entities and relations a tenant declares."""

from pydantic import BaseModel, ConfigDict, Field


class EntityType(BaseModel):
    """One kind of record a tenant's data contains, declared by name in a tenant's config."""

    model_config = ConfigDict(extra="forbid")

    name: str
    key: str
    fields: list[str]


class RelationType(BaseModel):
    """A named, directed link between two entity types declared in a tenant's config."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str
    from_entity: str = Field(alias="from")
    to_entity: str = Field(alias="to")


class TenantOntology(BaseModel):
    """The full set of entity types and relations a tenant declares."""

    model_config = ConfigDict(extra="forbid")

    entities: list[EntityType]
    relations: list[RelationType] = Field(default_factory=list)
