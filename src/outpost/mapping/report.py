"""Mapping outcomes.

MappingReport itself is added in phase 2, once the csv/pdf connectors and
field resolver exist to produce one.
"""

from enum import StrEnum


class MappingOutcome(StrEnum):
    """What happened when a tenant's raw field was resolved to an ontology field."""

    MAPPED = "mapped"
    NEEDS_REVIEW = "needs_review"
    UNMAPPED = "unmapped"
