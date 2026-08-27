"""Spaces requests out so a burst of calls does not hammer the endpoint.

An agent loop issues its calls as fast as it produces them, which is what
pushed both models past their declared p99 budget and timed one of them
out entirely. This enforces a minimum gap between requests instead.

Compose it outside the budget check, Paced(Budgeted(provider)), so the
budget still measures how long the provider took to answer rather than
how long this class chose to wait before asking.
"""

import time
from dataclasses import dataclass, field

from outpost.llm.base import Completion, Message, Provider, ToolSpec


@dataclass
class PacedProvider:
    inner: Provider
    min_interval_seconds: float
    _last_request_at: float | None = field(default=None, init=False)

    def complete(
        self, messages: list[Message], *, tools: list[ToolSpec] | None = None
    ) -> Completion:
        self._wait_for_slot()
        try:
            return self.inner.complete(messages, tools=tools)
        finally:
            # Stamped after the call, so the gap is measured between the end
            # of one request and the start of the next. Spacing from the
            # start instead would let a slow call collapse the interval to
            # nothing.
            self._last_request_at = time.monotonic()

    def _wait_for_slot(self) -> None:
        if self._last_request_at is None:
            return
        remaining = self.min_interval_seconds - (time.monotonic() - self._last_request_at)
        if remaining > 0:
            time.sleep(remaining)
