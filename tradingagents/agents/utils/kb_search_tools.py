from typing import Annotated, Optional

from langchain_core.tools import tool

from tradingagents.dataflows import knowledge_base


@tool
def search_knowledge_base(
    query: Annotated[
        str,
        "The research question to ask, phrased in the language the research is "
        "written in (Vietnamese for the Vietnamese market)",
    ],
    ticker: Annotated[
        Optional[str],
        "Restrict to research tagged with this ticker; omit for macro, strategy "
        "and sector questions",
    ] = None,
) -> str:
    """Ask our own research library a question.

    Semantic search over this deployment's embedded research reports, on the
    question you write rather than a canned one. Ask several narrow questions —
    company outlook, sector, macro — rather than one broad one, and rephrase a
    question that comes back empty.

    The library holds undated research, not headlines: treat what it returns as
    background and use the news tools for dated events. A miss is reported as a
    miss; nothing is silently substituted from the open web.
    """
    return knowledge_base.search(query, ticker=ticker)
