"""Typed errors for the retrieval layer."""


class RetrievalError(Exception):
    """Base class for all retrieval errors."""


class EmbeddingCacheMissError(RetrievalError):
    """Raised when a piece of text has no cached embedding.

    Tests and CI must never call the network, so a cache miss during a
    test run is a real failure to fix (regenerate the fixture cache), not
    something to fall back from silently.
    """

    def __init__(self, *, text: str, input_type: str) -> None:
        self.text = text
        self.input_type = input_type
        preview = text[:40] + ("..." if len(text) > 40 else "")
        super().__init__(f"no cached {input_type} embedding for text: {preview!r}")
