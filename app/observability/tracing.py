from contextlib import contextmanager
from typing import Iterator, Optional

from opentelemetry import trace

from app.config import settings

tracer = trace.get_tracer("vaultrag")  # vendor-neutral OTel tracer, always active

# Langfuse client is optional: only constructed when both keys are configured, so the
# app runs (and stays silent) in dev/CI environments without a Langfuse instance —
# same guard pattern as guards.py's Guardrails AI client.
_langfuse_client = None
_langfuse_init_attempted = False


def get_langfuse_client():
    """Lazily construct the Langfuse client once; returns None (no-op) if unconfigured or unavailable."""
    global _langfuse_client, _langfuse_init_attempted
    if _langfuse_init_attempted:
        return _langfuse_client
    _langfuse_init_attempted = True

    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None  # tracing is opt-in — no keys means no Langfuse calls anywhere

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception:
        # Import or connection failure must never break the request path — tracing is
        # observability, not a hard dependency.
        _langfuse_client = None
    return _langfuse_client


@contextmanager
def span(name: str) -> Iterator[object]:
    """Create a vendor-neutral span for an operation."""
    with tracer.start_as_current_span(name) as current_span:
        yield current_span


@contextmanager
def traced_generation(trace_obj, name: str, **kwargs):
    """Wrap a Langfuse generation span; no-ops cleanly when trace_obj is None (Langfuse disabled)."""
    if trace_obj is None:
        yield None
        return
    generation = trace_obj.generation(name=name, **kwargs)
    try:
        yield generation
    finally:
        generation.end()


def start_trace(name: str, input_data: Optional[dict] = None):
    """Start a top-level Langfuse trace for one request; returns None when Langfuse is disabled."""
    client = get_langfuse_client()
    if client is None:
        return None
    return client.trace(name=name, input=input_data)
