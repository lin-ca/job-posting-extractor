"""Connector protocol for external service integrations."""

from typing import Any, Protocol, runtime_checkable

from job_posting_extractor.models import RawExtractionResult


@runtime_checkable
class Connector(Protocol):
    """
    Base protocol for all external service connectors.

    Defines lifecycle methods that all connectors must implement.
    Uses structural subtyping - any class implementing these methods
    satisfies the protocol without explicit inheritance.
    """

    async def initialize(self) -> None:
        """Initialize connector resources (connection pools, clients, etc.)."""
        ...

    async def cleanup(self) -> None:
        """Cleanup connector resources on shutdown."""
        ...

    async def health_check(self) -> dict[str, Any]:
        """Check connector health and return status information."""
        ...


@runtime_checkable
class JobExtractor(Connector, Protocol):
    """
    Protocol for connectors that can extract job postings.

    Extends Connector with job extraction capability.
    """

    async def extract_job_posting(self, job_text: str) -> RawExtractionResult:
        """Extract structured job posting data from text."""
        ...
