"""A provider that replays recorded request/response pairs instead of
calling a live api. The only provider CI ever constructs.

Fixtures are keyed by a hash of the normalized request (model, messages,
tools), the same pattern the retrieval embedding cache uses, so the same
call always resolves to the same fixture file regardless of which test
triggers it.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from outpost.llm.base import Completion, Message, ToolCall, ToolSpec, Usage
from outpost.llm.errors import ProviderError


def request_key(model: str, messages: list[Message], tools: list[ToolSpec] | None) -> str:
    # tool_call_id is excluded deliberately: it is an opaque id the
    # provider assigns per call, not part of the request's meaning, so
    # two semantically identical requests must not hash differently just
    # because a live provider happened to mint a different id each time.
    normalized = {
        "model": model,
        "messages": [
            {
                "role": message.role,
                "content": message.content,
                "name": message.name,
                "tool_calls": [
                    {"name": call.name, "arguments": call.arguments}
                    for call in (message.tool_calls or [])
                ],
            }
            for message in messages
        ],
        "tools": [{"name": tool.name, "parameters": tool.parameters} for tool in (tools or [])],
    }
    return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()


@dataclass
class RecordedProvider:
    fixtures_dir: Path
    model: str = "recorded"

    def complete(
        self, messages: list[Message], *, tools: list[ToolSpec] | None = None
    ) -> Completion:
        key = request_key(self.model, messages, tools)
        fixture_path = self.fixtures_dir / f"{key}.json"
        if not fixture_path.exists():
            raise ProviderError(
                model=self.model,
                detail=f"no recorded fixture for this request (expected {fixture_path.name})",
            )

        response = json.loads(fixture_path.read_text())["response"]
        return Completion(
            content=response.get("content"),
            tool_calls=[
                ToolCall(id=call["id"], name=call["name"], arguments=call["arguments"])
                for call in response.get("tool_calls", [])
            ],
            usage=Usage(**response.get("usage", {"prompt_tokens": 0, "completion_tokens": 0})),
            model=self.model,
        )
