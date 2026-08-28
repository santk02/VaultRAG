from contextlib import contextmanager
from typing import Iterator

from opentelemetry import trace

tracer = trace.get_tracer("vaultrag")


@contextmanager
def span(name: str) -> Iterator[object]:
    """Create a vendor-neutral span for an operation."""
    with tracer.start_as_current_span(name) as current_span:
        yield current_span
