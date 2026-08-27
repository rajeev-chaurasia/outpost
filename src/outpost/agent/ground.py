"""Citation binding.

The logic that binds claims in an answer to spans is added in phase 4.
"""

from pydantic import BaseModel, ConfigDict

from outpost.retrieval.document import Span


class Citation(BaseModel):
    """One assertion in an answer, bound to the exact source text it came from."""

    model_config = ConfigDict(extra="forbid")

    assertion: str
    span: Span
