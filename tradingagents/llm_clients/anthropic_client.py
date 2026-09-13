import re
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

_PROMPT_CACHE_CONTROL = {"type": "ephemeral"}


def _cache_static_system_prefix(input_: Any) -> Any:
    """Mark the first system message as Anthropic's reusable cache prefix.

    Agent prompts place immutable instructions in their first system message
    and dates, instrument data, and tool results later. The explicit marker
    caches only that stable prefix instead of paying cache-write pricing for
    every run-specific suffix.
    """
    messages = (
        input_
        if isinstance(input_, list)
        else (input_.to_messages() if hasattr(input_, "to_messages") else None)
    )
    if not messages:
        return input_

    for index, message in enumerate(messages):
        if not isinstance(message, SystemMessage) or not message.content:
            continue
        if isinstance(message.content, str):
            content = [
                {
                    "type": "text",
                    "text": message.content,
                    "cache_control": _PROMPT_CACHE_CONTROL.copy(),
                }
            ]
        elif isinstance(message.content, list):
            content = list(message.content)
            last = content[-1]
            if isinstance(last, str):
                content[-1] = {
                    "type": "text",
                    "text": last,
                    "cache_control": _PROMPT_CACHE_CONTROL.copy(),
                }
            elif isinstance(last, dict):
                content[-1] = {
                    **last,
                    "cache_control": _PROMPT_CACHE_CONTROL.copy(),
                }
            else:
                continue
        else:
            continue
        return [
            *messages[:index],
            message.model_copy(update={"content": content}),
            *messages[index + 1 :],
        ]
    return input_


_PASSTHROUGH_KWARGS = (
    "timeout",
    "max_retries",
    "api_key",
    "max_tokens",
    "temperature",
    "callbacks",
    "http_client",
    "http_async_client",
    "effort",
)

# Anthropic's extended-thinking ``effort`` parameter is accepted by Opus 4.5+,
# Sonnet 4.6+, and the Claude 5 family (Sonnet 5, Fable 5). Sonnet 4.5 and any
# Haiku version 400 with ``"This model does not support the effort parameter"``
# (#831). Versions may be dotted (``opus-4-8``) or single-number (``sonnet-5``,
# ``fable-5``); the per-family minimum below is forward-compatible.
_EFFORT_EXACT = {
    "claude-mythos-preview",  # non-standard preview name; effort-capable
    "claude-mythos-5",  # Fable 5 twin (Project Glasswing); effort-capable
}
_EFFORT_MODEL = re.compile(r"^claude-(opus|sonnet|fable)-(\d+)(?:-(\d+))?$")
_EFFORT_MIN_VERSION = {"opus": (4, 5), "sonnet": (4, 6), "fable": (5, 0)}


def _supports_effort(model: str) -> bool:
    """Whether Anthropic accepts the ``effort`` parameter for this model."""
    model_lc = model.lower()
    if model_lc in _EFFORT_EXACT:
        return True
    match = _EFFORT_MODEL.match(model_lc)
    if not match:
        return False
    family = match.group(1)
    major = int(match.group(2))
    minor = int(match.group(3)) if match.group(3) else 0
    return (major, minor) >= _EFFORT_MIN_VERSION[family]


class NormalizedChatAnthropic(ChatAnthropic):
    """ChatAnthropic with normalized content output.

    Claude models with extended thinking or tool use return content as a
    list of typed blocks. This normalizes to string for consistent
    downstream handling.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(
            super().invoke(_cache_static_system_prefix(input), config, **kwargs)
        )


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models."""

    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatAnthropic instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in _PASSTHROUGH_KWARGS:
            if key not in self.kwargs:
                continue
            if key == "effort" and not _supports_effort(self.model):
                continue
            llm_kwargs[key] = self.kwargs[key]

        return NormalizedChatAnthropic(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Anthropic."""
        return validate_model("anthropic", self.model)
