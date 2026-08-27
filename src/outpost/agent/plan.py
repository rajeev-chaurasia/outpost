"""Turns a request into an ordered sequence of tool calls and a final
answer.

The provider is handed its tools directly and decides for itself when to
call one; this is a bounded loop around that decision, not a separate
planning step and not a graph framework. Simpler to explain, and there
is nothing here a graph would do better.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from outpost.agent.tools.base import Tool
from outpost.llm.base import Completion, Message, Provider


@dataclass(frozen=True)
class Step:
    tool_name: str
    arguments: dict[str, Any]
    result: Any


@dataclass(frozen=True)
class PlanResult:
    steps: list[Step]
    final_content: str | None
    completions: list[Completion] = field(default_factory=list)


def run(
    provider: Provider,
    tools: dict[str, Tool],
    *,
    system_prompt: str,
    user_request: str,
    max_steps: int = 6,
) -> PlanResult:
    messages: list[Message] = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_request),
    ]
    tool_specs = [tool.spec for tool in tools.values()]
    steps: list[Step] = []
    completions: list[Completion] = []

    for _ in range(max_steps):
        completion = provider.complete(messages, tools=tool_specs)
        completions.append(completion)

        if not completion.tool_calls:
            return PlanResult(
                steps=steps, final_content=completion.content, completions=completions
            )

        messages.append(
            Message(role="assistant", content=completion.content, tool_calls=completion.tool_calls)
        )
        for call in completion.tool_calls:
            tool = tools.get(call.name)
            result: Any = (
                {"error": f"unknown tool {call.name!r}"}
                if tool is None
                else tool.invoke(call.arguments)
            )
            steps.append(Step(tool_name=call.name, arguments=call.arguments, result=result))
            messages.append(
                Message(
                    role="tool",
                    content=json.dumps(result, default=str),
                    tool_call_id=call.id,
                    name=call.name,
                )
            )

    return PlanResult(steps=steps, final_content=None, completions=completions)
