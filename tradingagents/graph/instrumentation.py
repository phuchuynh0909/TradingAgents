# TradingAgents/graph/instrumentation.py
"""Per-node progress and failure logging for a graph run.

A run that dies deep in the graph otherwise surfaces as a bare stack trace: the
driver knows the analysis failed, but not which of the dozen-odd agents was
holding the ball, nor how far the run had got. This handler answers both, and
times every node on the way through.

It is a callback handler rather than a wrapper around the node callables, which
is not a stylistic choice. Nodes come in three shapes — a plain function
(``market_analyst_node``), a ``functools.partial`` (``trader_node``) and a
``ToolNode``, which is a Runnable and must be invoked rather than called — so no
single wrapper calls all three correctly. Worse, LangGraph decides whether to
hand a node its ``config`` by inspecting the callable's signature, so wrapping
severs config propagation for the tool nodes and with it their tracing.

LangGraph stamps ``langgraph_node`` and ``langgraph_step`` into the metadata of
every node run, which is what makes the callback route work: those keys are both
the node's identity and the means of telling a real node apart from the graph's
own root run and from the nested LLM/prompt chains underneath it.

Attach it through the invocation config, which ``Propagator.get_graph_args``
already exposes::

    tracker = NodeProgressLogger()
    args = propagator.get_graph_args(callbacks=[tracker])
    for chunk in graph.stream(state, **args):
        ...
    if tracker.failed_node:
        ...

Every node start and finish is logged at INFO and a failure at ERROR, so setting
this package's log level to WARNING keeps the failures and drops the running
commentary.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)

# Written into every node run's metadata by LangGraph (pregel/_algo.py).
_NODE_KEY = "langgraph_node"
_STEP_KEY = "langgraph_step"


class NodeProgressLogger(BaseCallbackHandler):
    """Logs each graph node's start, duration and failure; remembers the failure.

    Safe to attach to any run: every hook ignores anything it has no record of,
    so a handler added to a run already in progress, or one whose start it never
    saw, degrades to logging less rather than raising. That matters because a
    handler that throws costs a warning per node from LangChain's dispatcher.

    Thread-safe. LangGraph runs nodes on a pool, so two can be open at once, and
    the bookkeeping is keyed by run id rather than by "the current node".
    """

    def __init__(self, log: logging.Logger | None = None) -> None:
        self._log = log or logger
        # run id -> (node name, perf_counter at start)
        self._open: dict[UUID, tuple[str, float]] = {}
        self._durations: dict[str, float] = {}
        self._lock = threading.Lock()

        #: ``(node, duration_ms)`` in completion order.
        self.completed: list[tuple[str, float]] = []
        #: The node that raised, or None. First failure wins — in a parallel
        #: step a second node can fail while the first is already unwinding, and
        #: the one that started the collapse is the informative one.
        self.failed_node: str | None = None

    # -- hooks ---------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        node = self._node_of(metadata)
        if node is None:
            # The graph's own root run, or a nested chain inside a node. Neither
            # is a step, and logging them would bury the ones that are.
            return
        with self._lock:
            self._open[run_id] = (node, time.perf_counter())
        step = (metadata or {}).get(_STEP_KEY, "?")
        self._log.info("node %s: start (step %s)", node, step)

    def on_chain_end(
        self, outputs: Any, *, run_id: UUID | None = None, **kwargs: Any
    ) -> None:
        node, elapsed = self._close(run_id)
        if node is None:
            return
        with self._lock:
            self._durations[node] = elapsed
            self.completed.append((node, elapsed))
        self._log.info("node %s: ok in %.0f ms", node, elapsed)

    def on_chain_error(
        self, error: BaseException, *, run_id: UUID | None = None, **kwargs: Any
    ) -> None:
        node, elapsed = self._close(run_id)
        if node is None:
            return

        # A client disconnect closes the stream, which raises GeneratorExit into
        # the Pregel loop, and LangGraph reports any BaseException through this
        # hook. Nothing broke — the run was abandoned — so it must not be
        # recorded as a node failure or every cancelled run reads as a crash.
        if isinstance(error, GeneratorExit):
            self._log.info("node %s: cancelled after %.0f ms", node, elapsed)
            return

        if self.failed_node is None:
            self.failed_node = node
        self._log.error(
            "node %s: failed after %.0f ms — %s: %s",
            node,
            elapsed,
            type(error).__name__,
            error,
        )

    # -- readers -------------------------------------------------------------

    def duration_ms(self, node: str) -> float | None:
        """How long ``node`` last took, or None if it has not finished one."""
        with self._lock:
            return self._durations.get(node)

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _node_of(metadata: dict[str, Any] | None) -> str | None:
        if not isinstance(metadata, dict):
            return None
        node = metadata.get(_NODE_KEY)
        return str(node) if node else None

    def _close(self, run_id: UUID | None) -> tuple[str | None, float]:
        with self._lock:
            entry = self._open.pop(run_id, None)
        if entry is None:
            return None, 0.0
        node, started = entry
        return node, (time.perf_counter() - started) * 1000
