"""FastAPI dependencies for dependency injection."""

from typing import Annotated

from fastapi import Depends, Request

from job_posting_extractor.config import Settings, get_settings
from job_posting_extractor.connectors.base import JobExtractor
from job_posting_extractor.services.extraction import ExtractionService


def get_connector(request: Request) -> JobExtractor:
    """Get the lifespan-managed connector from app state."""
    connector: JobExtractor = request.app.state.claude_connector
    return connector


def get_extraction_service(
    connector: Annotated[JobExtractor, Depends(get_connector)],
) -> ExtractionService:
    """Get extraction service with injected dependencies."""
    return ExtractionService(connector=connector)


# Type aliases for cleaner route signatures
SettingsDep = Annotated[Settings, Depends(get_settings)]
ConnectorDep = Annotated[JobExtractor, Depends(get_connector)]
ExtractionServiceDep = Annotated[ExtractionService, Depends(get_extraction_service)]
