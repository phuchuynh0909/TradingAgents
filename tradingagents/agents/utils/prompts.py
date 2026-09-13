"""Prompt assembly that preserves reusable LLM cache prefixes."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

TOOL_AGENT_RUNTIME_CONTEXT = (
    "You are a helpful AI assistant, collaborating with other assistants."
    " Use the provided tools to progress towards answering the question."
    " If you are unable to fully answer, that's OK; another assistant with different tools"
    " will help where you left off. Execute what you can to make progress."
    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
    " You have access to the following tools: {tool_names}."
    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges."
    " {instrument_context}"
)


def build_cache_friendly_prompt(
    static_instructions: str, runtime_context: str
) -> ChatPromptTemplate:
    """Build a prompt with immutable instructions ahead of run-specific context.

    LLM providers reuse only an exact prompt prefix. Keeping the agent's long,
    static instructions in their own first system message allows OpenAI's
    automatic cache and Anthropic's explicit cache marker to reuse that work;
    dates, instrument identity, tool names, and fetched data remain a suffix.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", static_instructions),
            ("system", runtime_context),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
