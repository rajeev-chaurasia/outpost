"""Planner done-tests: a no-tool-call completion ends the loop with the
final answer, a tool call gets executed and fed back as a tool message,
an unknown tool name does not crash, and the loop is bounded.
"""

from dataclasses import dataclass, field

from outpost.agent.plan import run
from outpost.agent.tools.base import Tool
from outpost.llm.base import Completion, Message, ToolCall, ToolSpec, Usage


@dataclass
class FakeProvider:
    """Returns one scripted completion per call, in order."""

    completions: list[Completion]
    seen_messages: list[list[Message]] = field(default_factory=list)

    def complete(
        self, messages: list[Message], *, tools: list[ToolSpec] | None = None
    ) -> Completion:
        self.seen_messages.append(list(messages))
        return self.completions[len(self.seen_messages) - 1]


@dataclass
class FakeTool:
    name: str
    result: object

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description="d", parameters={"type": "object"})

    def invoke(self, arguments: dict[str, object]) -> object:
        return self.result


def _usage() -> Usage:
    return Usage(prompt_tokens=1, completion_tokens=1)


def test_no_tool_call_ends_the_loop_with_final_content() -> None:
    provider = FakeProvider(
        completions=[Completion(content="the answer", tool_calls=[], usage=_usage(), model="m")]
    )
    result = run(provider, tools={}, system_prompt="s", user_request="u")

    assert result.final_content == "the answer"
    assert result.steps == []


def test_tool_call_is_executed_and_recorded_as_a_step() -> None:
    tool = FakeTool(name="search", result={"hits": 1})
    provider = FakeProvider(
        completions=[
            Completion(
                content=None,
                tool_calls=[ToolCall(id="c1", name="search", arguments={"query": "x"})],
                usage=_usage(),
                model="m",
            ),
            Completion(content="done", tool_calls=[], usage=_usage(), model="m"),
        ]
    )

    result = run(provider, tools={"search": tool}, system_prompt="s", user_request="u")

    assert result.final_content == "done"
    assert len(result.steps) == 1
    assert result.steps[0].tool_name == "search"
    assert result.steps[0].result == {"hits": 1}

    # the tool result must reach the provider as a tool message on the next turn
    second_call_messages = provider.seen_messages[1]
    assert any(m.role == "tool" and m.tool_call_id == "c1" for m in second_call_messages)


def test_unknown_tool_name_does_not_raise() -> None:
    provider = FakeProvider(
        completions=[
            Completion(
                content=None,
                tool_calls=[ToolCall(id="c1", name="does_not_exist", arguments={})],
                usage=_usage(),
                model="m",
            ),
            Completion(content="done", tool_calls=[], usage=_usage(), model="m"),
        ]
    )

    result = run(provider, tools={}, system_prompt="s", user_request="u")

    assert result.steps[0].result == {"error": "unknown tool 'does_not_exist'"}


def test_loop_is_bounded_by_max_steps() -> None:
    always_calls_tool = Completion(
        content=None,
        tool_calls=[ToolCall(id="c1", name="search", arguments={"query": "x"})],
        usage=_usage(),
        model="m",
    )
    provider = FakeProvider(completions=[always_calls_tool] * 10)
    tool: Tool = FakeTool(name="search", result={"hits": 1})

    result = run(provider, tools={"search": tool}, system_prompt="s", user_request="u", max_steps=3)

    assert result.final_content is None
    assert len(result.steps) == 3
    assert len(provider.seen_messages) == 3
