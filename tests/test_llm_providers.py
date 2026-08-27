"""Provider done-tests: the recorded provider replays a fixture keyed by
a hash of the normalized request, and raises a typed error rather than
guessing when no fixture matches; the wire-format helpers round-trip a
tool call correctly.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from outpost.llm.base import Message, ToolCall, ToolSpec
from outpost.llm.errors import ProviderError
from outpost.llm.openai_compatible import _message_to_wire, _parse_completion, _tool_to_wire
from outpost.llm.recorded import RecordedProvider, request_key


def _write_fixture(fixtures_dir: Path, key: str, response: dict[str, Any]) -> None:
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    (fixtures_dir / f"{key}.json").write_text(json.dumps({"response": response}))


def test_recorded_provider_replays_matching_fixture(tmp_path: Path) -> None:
    messages = [Message(role="user", content="hello")]
    key = request_key("recorded", messages, None)
    _write_fixture(
        tmp_path,
        key,
        {
            "content": "hi there",
            "tool_calls": [],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        },
    )

    provider = RecordedProvider(fixtures_dir=tmp_path)
    completion = provider.complete(messages)

    assert completion.content == "hi there"
    assert completion.tool_calls == []
    assert completion.usage.prompt_tokens == 3


def test_recorded_provider_replays_a_tool_call(tmp_path: Path) -> None:
    messages = [Message(role="user", content="search for something")]
    tools = [ToolSpec(name="search", description="d", parameters={"type": "object"})]
    key = request_key("recorded", messages, tools)
    _write_fixture(
        tmp_path,
        key,
        {
            "content": None,
            "tool_calls": [{"id": "call-1", "name": "search", "arguments": {"query": "invoice"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 4},
        },
    )

    completion = RecordedProvider(fixtures_dir=tmp_path).complete(messages, tools=tools)

    assert completion.tool_calls == [
        ToolCall(id="call-1", name="search", arguments={"query": "invoice"})
    ]


def test_recorded_provider_raises_on_missing_fixture(tmp_path: Path) -> None:
    with pytest.raises(ProviderError):
        RecordedProvider(fixtures_dir=tmp_path).complete([Message(role="user", content="anything")])


def test_request_key_changes_with_message_history() -> None:
    a = request_key("m", [Message(role="user", content="one")], None)
    b = request_key("m", [Message(role="user", content="two")], None)
    assert a != b


def test_request_key_is_stable_for_identical_requests() -> None:
    messages = [Message(role="user", content="same")]
    assert request_key("m", messages, None) == request_key("m", messages, None)


def test_message_to_wire_includes_tool_calls_when_present() -> None:
    message = Message(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id="c1", name="search", arguments={"query": "x"})],
    )
    wire = _message_to_wire(message)
    assert wire["tool_calls"][0]["function"]["name"] == "search"
    assert json.loads(wire["tool_calls"][0]["function"]["arguments"]) == {"query": "x"}


def test_tool_to_wire_shape() -> None:
    tool = ToolSpec(name="search", description="d", parameters={"type": "object"})
    wire = _tool_to_wire(tool)
    assert wire == {
        "type": "function",
        "function": {"name": "search", "description": "d", "parameters": {"type": "object"}},
    }


def test_parse_completion_raises_on_unexpected_shape() -> None:
    with pytest.raises(ProviderError):
        _parse_completion({}, "some-model")


def test_parse_completion_raises_on_malformed_tool_call_arguments() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "search", "arguments": "{not json"}}
                    ],
                }
            }
        ]
    }
    with pytest.raises(ProviderError):
        _parse_completion(payload, "some-model")
