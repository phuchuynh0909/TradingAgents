"""Per-node progress and failure logging for a graph run.

LangGraph stamps ``langgraph_node``/``langgraph_step`` into the metadata of every
node run (pregel/_algo.py), which is what lets a plain callback handler report
which node is running, how long it took and which one failed — without wrapping
the node callables, whose three shapes (plain function, functools.partial and
ToolNode) cannot be called through one uniform wrapper.
"""

import logging
import unittest
import uuid

from tradingagents.graph.instrumentation import NodeProgressLogger

LOGGER_NAME = "tradingagents.graph.instrumentation"


def _node_meta(name, step=1):
    return {"langgraph_node": name, "langgraph_step": step}


class NodeProgressLoggerTests(unittest.TestCase):
    def setUp(self):
        self.handler = NodeProgressLogger()

    def _run_node(self, name, run_id=None, step=1):
        run_id = run_id or uuid.uuid4()
        self.handler.on_chain_start({}, {}, run_id=run_id, metadata=_node_meta(name, step))
        return run_id

    def test_a_completed_node_logs_its_name_and_duration(self):
        with self.assertLogs(LOGGER_NAME, level="INFO") as captured:
            run_id = self._run_node("Market Analyst")
            self.handler.on_chain_end({}, run_id=run_id)

        joined = "\n".join(captured.output)
        self.assertIn("Market Analyst", joined)
        self.assertIn("start", joined)
        self.assertRegex(joined, r"ok in \d+ ms")

    def test_a_failed_node_is_named_and_recorded(self):
        with self.assertLogs(LOGGER_NAME, level="INFO") as captured:
            run_id = self._run_node("Bull Researcher")
            self.handler.on_chain_error(ValueError("model refused"), run_id=run_id)

        joined = "\n".join(captured.output)
        self.assertIn("Bull Researcher", joined)
        self.assertIn("ValueError", joined)
        self.assertIn("model refused", joined)
        # The runner reads this to name the node in its error event and in the
        # LangSmith tag, rather than scraping the log it just wrote.
        self.assertEqual(self.handler.failed_node, "Bull Researcher")

    def test_the_root_graph_run_is_not_treated_as_a_node(self):
        # The graph's own run has no langgraph_node metadata. Without this filter
        # every run would log a bogus start/end pair around the real nodes.
        root = uuid.uuid4()
        self.handler.on_chain_start({}, {}, run_id=root, metadata={"thread_id": "x"})
        self.handler.on_chain_end({}, run_id=root)

        self.assertIsNone(self.handler.failed_node)
        self.assertEqual(self.handler.completed, [])

    def test_cancellation_is_not_recorded_as_a_node_failure(self):
        # Closing the stream on a client disconnect raises GeneratorExit into the
        # Pregel loop, and LangGraph reports it through on_chain_error like any
        # other BaseException. It is a cancellation, not a node that broke.
        run_id = self._run_node("Trader")
        self.handler.on_chain_error(GeneratorExit(), run_id=run_id)

        self.assertIsNone(self.handler.failed_node)

    def test_interleaved_nodes_are_tracked_by_run_id(self):
        # Nodes run on a thread pool, so two can be open at once.
        first = self._run_node("News Analyst", step=1)
        second = self._run_node("Fundamentals Analyst", step=2)

        with self.assertLogs(LOGGER_NAME, level="INFO") as captured:
            self.handler.on_chain_end({}, run_id=second)
            self.handler.on_chain_end({}, run_id=first)

        joined = "\n".join(captured.output)
        self.assertIn("Fundamentals Analyst: ok", joined)
        self.assertIn("News Analyst: ok", joined)
        self.assertEqual(
            [name for name, _ in self.handler.completed],
            ["Fundamentals Analyst", "News Analyst"],
        )

    def test_duration_is_available_per_node(self):
        run_id = self._run_node("Research Manager")
        self.handler.on_chain_end({}, run_id=run_id)

        self.assertIsNotNone(self.handler.duration_ms("Research Manager"))
        self.assertIsNone(self.handler.duration_ms("Never Ran"))

    def test_an_end_without_a_start_is_ignored(self):
        # Handlers can be attached to a run already in progress; a stray end must
        # not raise out of the callback and take the graph run with it.
        self.handler.on_chain_end({}, run_id=uuid.uuid4())
        self.handler.on_chain_error(ValueError("x"), run_id=uuid.uuid4())

        self.assertEqual(self.handler.completed, [])
        self.assertIsNone(self.handler.failed_node)

    def test_the_handler_never_raises_into_the_graph(self):
        # LangChain's handle_event catches handler errors, but a handler that
        # throws still costs a warning per node. Missing metadata is normal.
        self.handler.on_chain_start({}, {}, run_id=uuid.uuid4(), metadata=None)
        self.assertEqual(self.handler.completed, [])


if __name__ == "__main__":
    unittest.main()
