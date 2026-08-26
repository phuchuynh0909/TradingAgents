"""The knowledge-base tool the news analyst drives with its own question.

``get_news``/``get_global_news`` also read the knowledge base, but as tier 1 of a
tiered tool firing one canned query. ``search_knowledge_base`` takes the
analyst's own question instead, so the wiring that has to hold is: the tool is
advertised in the prompt AND executable in the news ToolNode (bound-but-missing
makes the model report it "unavailable"), and its implementation is repointable
by the host platform — the ``market_data_validator.load_ohlcv`` pattern.
"""
import inspect

import pytest

import tradingagents.agents.analysts.news_analyst as na
import tradingagents.dataflows.knowledge_base as kb
from tradingagents.agents.utils.kb_search_tools import search_knowledge_base
from tradingagents.graph.trading_graph import TradingAgentsGraph


@pytest.mark.unit
def test_tool_takes_a_free_text_query_and_an_optional_ticker():
    args = search_knowledge_base.args
    assert "query" in args
    assert "ticker" in args


@pytest.mark.unit
def test_tool_delegates_to_the_repointable_implementation(monkeypatch):
    seen = {}

    def fake(query, ticker=None):
        seen.update(query=query, ticker=ticker)
        return "# Knowledge-base research for HPG"

    monkeypatch.setattr(kb, "search_kb", fake)

    out = search_knowledge_base.invoke(
        {"query": "triển vọng ngành thép", "ticker": "HPG"}
    )

    assert seen == {"query": "triển vọng ngành thép", "ticker": "HPG"}
    assert "Knowledge-base research for HPG" in out


@pytest.mark.unit
def test_no_registered_knowledge_base_returns_a_sentinel(monkeypatch):
    monkeypatch.setattr(kb, "search_kb", None)

    out = search_knowledge_base.invoke({"query": "chính sách tiền tệ của NHNN"})

    assert out.startswith("KB_UNAVAILABLE")


@pytest.mark.unit
def test_news_toolnode_can_execute_the_kb_search():
    # _create_tool_nodes does not use self -> call unbound (avoids building LLMs).
    nodes = TradingAgentsGraph._create_tool_nodes(None)
    news_tools = set(nodes["news"].tools_by_name)
    assert "search_knowledge_base" in news_tools, (
        "search_knowledge_base is advertised to the news analyst but not "
        "registered in the news ToolNode, so the model's call fails."
    )
    assert {"get_news", "get_global_news", "get_macro_indicators"} <= news_tools


@pytest.mark.unit
def test_news_prompt_matches_the_kb_tool_signature():
    src = inspect.getsource(na)
    assert "search_knowledge_base(query, ticker=None)" in src
