"""An OpenAI-compatible chat completions provider.

Points at build.nvidia.com by default, but the model and base url are
both parameters rather than hardcoded, since the wire format is the same
against any OpenAI-compatible endpoint.
"""

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx

from outpost.llm.base import Completion, Message, ToolCall, ToolSpec, Usage
from outpost.llm.errors import ProviderError

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


@dataclass
class OpenAICompatibleProvider:
    model: str
    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = 30.0

    def complete(
        self, messages: list[Message], *, tools: list[ToolSpec] | None = None
    ) -> Completion:
        api_key = self.api_key or os.environ["LLM_API_KEY"]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_wire(message) for message in messages],
            "temperature": 0,
        }
        if tools:
            payload["tools"] = [_tool_to_wire(tool) for tool in tools]

        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(model=self.model, detail=str(exc)) from exc

        return _parse_completion(response.json(), self.model)


def _message_to_wire(message: Message) -> dict[str, Any]:
    wire: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_call_id is not None:
        wire["tool_call_id"] = message.tool_call_id
    if message.name is not None:
        wire["name"] = message.name
    if message.tool_calls:
        wire["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    return wire


def _tool_to_wire(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _parse_completion(payload: dict[str, Any], model: str) -> Completion:
    try:
        message = payload["choices"][0]["message"]
        usage_raw = payload.get("usage", {})
    except (KeyError, IndexError) as exc:
        raise ProviderError(model=model, detail=f"unexpected response shape: {exc}") from exc

    tool_calls = []
    for raw_call in message.get("tool_calls") or []:
        try:
            arguments = json.loads(raw_call["function"]["arguments"])
        except json.JSONDecodeError as exc:
            raise ProviderError(
                model=model, detail=f"malformed tool call arguments: {exc}"
            ) from exc
        tool_calls.append(
            ToolCall(id=raw_call["id"], name=raw_call["function"]["name"], arguments=arguments)
        )

    return Completion(
        content=message.get("content"),
        tool_calls=tool_calls,
        usage=Usage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
        ),
        model=model,
    )
