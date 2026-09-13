"""Prompt-cache invariants for reusable agent instructions."""

from unittest.mock import patch

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tradingagents.agents.utils.prompts import build_cache_friendly_prompt
from tradingagents.llm_clients.anthropic_client import (
    NormalizedChatAnthropic,
    _cache_static_system_prefix,
)


def test_prompt_keeps_static_instructions_before_runtime_context():
    prompt = build_cache_friendly_prompt(
        "Static analyst instructions.", "Run date: {current_date}."
    )

    messages = prompt.format_messages(
        current_date="2026-09-08", messages=[HumanMessage(content="Analyze HPG")]
    )

    assert [message.type for message in messages] == ["system", "system", "human"]
    assert messages[0].content == "Static analyst instructions."
    assert messages[1].content == "Run date: 2026-09-08."


def test_anthropic_marks_only_static_system_prefix_for_caching():
    static = SystemMessage(content="Stable analyst instructions.")
    runtime = SystemMessage(content="Ticker: HPG; date: 2026-09-08.")

    cached = _cache_static_system_prefix([static, runtime, HumanMessage(content="Proceed")])

    assert static.content == "Stable analyst instructions."
    assert cached[0].content == [
        {
            "type": "text",
            "text": "Stable analyst instructions.",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert cached[1] is runtime


def test_anthropic_client_transmits_cache_marked_prefix():
    llm = NormalizedChatAnthropic(model="claude-haiku-4-5", api_key="placeholder")

    with patch.object(ChatAnthropic, "invoke", return_value=AIMessage(content="ok")) as invoke:
        response = llm.invoke([SystemMessage(content="Stable"), HumanMessage(content="Analyze")])

    forwarded = invoke.call_args.args[0]
    assert response.content == "ok"
    assert forwarded[0].content[0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_payload_keeps_static_cache_breakpoint():
    llm = NormalizedChatAnthropic(model="claude-haiku-4-5", api_key="placeholder")

    payload = llm._get_request_payload(
        _cache_static_system_prefix(
            [SystemMessage(content="Stable"), HumanMessage(content="Analyze")]
        )
    )

    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}
