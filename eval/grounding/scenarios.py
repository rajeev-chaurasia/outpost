"""Shared scenario definitions for grounding measurement and fixture
generation, so the exact wording used to record a fixture is the same
wording used to replay it. A mismatch here is a cache miss, not a
subtly wrong answer.
"""

from dataclasses import dataclass

_DEALER_AR_SYSTEM_PROMPT = (
    "You are a helpful assistant for a dealership accounts receivable team. "
    "Use the search tool to find relevant statement text before answering. "
    "Answer only using information the search tool returns, in one or two "
    "short sentences."
)

_CLAIMS_INTAKE_SYSTEM_PROMPT = (
    "You are a helpful assistant for an insurance claims intake team. "
    "Use the search tool to find relevant policy text before answering. "
    "Answer only using information the search tool returns, in one or two "
    "short sentences."
)

_UTILITY_OPS_SYSTEM_PROMPT = (
    "You are a helpful assistant for a utility field operations team. "
    "Use the search tool to find relevant service agreement text before "
    "answering. Answer only using information the search tool returns, in "
    "one or two short sentences."
)


@dataclass(frozen=True)
class GroundingScenario:
    tenant_id: str
    system_prompt: str
    user_request: str


SCENARIOS = [
    GroundingScenario(
        tenant_id="dealer_ar",
        system_prompt=_DEALER_AR_SYSTEM_PROMPT,
        user_request="According to the account statements, was invoice INV-1001 paid, and how?",
    ),
    GroundingScenario(
        tenant_id="claims_intake",
        system_prompt=_CLAIMS_INTAKE_SYSTEM_PROMPT,
        user_request="What is the deductible on policy POL-500, and who is the policyholder?",
    ),
    GroundingScenario(
        tenant_id="utility_ops",
        system_prompt=_UTILITY_OPS_SYSTEM_PROMPT,
        user_request="What is the response time on the service agreement for account ACC-701?",
    ),
]
