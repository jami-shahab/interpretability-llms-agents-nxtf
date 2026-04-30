"""Langfuse tracing helpers — all functions are no-ops when client is None."""

from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional


@contextmanager
def sample_trace(
    client: Optional[Any],
    case_id: str,
    expected_decision: str,
    config_name: str,
    run_id: str,
) -> Generator[Any, None, None]:
    """Open a Langfuse trace for one pipeline run (no-op if client is None).

    Yields
    ------
    trace : Any
        Langfuse trace handle, or a sentinel object if tracing is inactive.
    """
    if client is None:
        yield _NoOpTrace()
        return

    trace = client.trace(
        name=f"sme_capital/{case_id}",
        input={"case_id": case_id, "expected_decision": expected_decision},
        metadata={"config": config_name, "run_id": run_id},
    )
    try:
        yield trace
    finally:
        pass  # Langfuse flushes automatically


def open_llm_span(
    trace: Optional[Any],
    name: str,
    input_data: dict,
    model: str,
    metadata: Optional[dict] = None,
) -> Optional[Any]:
    """Open a child generation span (no-op if trace is None or NoOp)."""
    if trace is None or isinstance(trace, _NoOpTrace):
        return None
    try:
        return trace.generation(
            name=name,
            input=input_data,
            model=model,
            metadata=metadata or {},
        )
    except Exception:
        return None


def close_span(span: Optional[Any], output: Optional[dict] = None) -> None:
    """Close a Langfuse span (no-op if span is None)."""
    if span is None:
        return
    try:
        span.end(output=output or {})
    except Exception:
        pass


def log_trace_scores(trace: Optional[Any], scores: Dict[str, float]) -> None:
    """Attach metric scores to a Langfuse trace (no-op if trace is None)."""
    if trace is None or isinstance(trace, _NoOpTrace):
        return
    for name, value in scores.items():
        try:
            trace.score(name=name, value=value)
        except Exception:
            pass


class _NoOpTrace:
    """Sentinel returned when Langfuse is not configured."""

    id: Optional[str] = None

    def update(self, **kwargs: Any) -> None:  # noqa: ANN401
        """No-op update."""

    def score(self, **kwargs: Any) -> None:  # noqa: ANN401
        """No-op score."""
