"""Knowledge-base retrieval, supplied by the host deployment.

Upstream ships no knowledge base, so ``search_kb`` is None here and the tool
answers with a sentinel. A deployment that *has* one — this fork's backend embeds
Vietnamese research reports into Qdrant — repoints this module global at its own
implementation during vendor registration. That is the same seam
``market_data_validator.load_ohlcv`` uses, and it keeps a corpus that has no
notion of competing vendors out of the ``route_to_vendor`` dispatch table.
"""
from __future__ import annotations

from typing import Callable, Optional

# Repointed at registration time; see the module docstring.
search_kb: Optional[Callable[..., str]] = None

KB_UNAVAILABLE = (
    "KB_UNAVAILABLE: no knowledge base is configured for this deployment. Use "
    "the news and macro tools instead; do not fabricate research findings."
)


def search(query: str, ticker: str | None = None) -> str:
    """Run one knowledge-base query, or say that there is no knowledge base."""
    if search_kb is None:
        return KB_UNAVAILABLE
    return search_kb(query, ticker=ticker)
