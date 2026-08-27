"""Typed errors for the ontology and tenant configuration layer."""


class OntologyError(Exception):
    """Base class for all ontology and tenant configuration errors."""


class ConfigValidationError(OntologyError):
    """Raised when a tenant config fails cross-reference validation.

    Carries the offending key and source line so the caller can point a
    tenant admin at the exact spot in the yaml file, rather than a pydantic
    stack trace.
    """

    def __init__(self, message: str, *, key: str, line: int | None) -> None:
        self.key = key
        self.line = line
        location = f"line {line}" if line is not None else "unknown line"
        super().__init__(f"{message} ({key}, {location})")
