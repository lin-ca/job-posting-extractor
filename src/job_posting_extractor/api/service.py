"""FastAPI application factory using composition pattern."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from job_posting_extractor import __version__
from job_posting_extractor.config import get_settings
from job_posting_extractor.connectors.base import JobExtractor
from job_posting_extractor.connectors.claude import ClaudeConnector
from job_posting_extractor.connectors.mock_claude import MockClaudeConnector
from job_posting_extractor.exceptions import BusinessError


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Lifespan context manager for startup/shutdown events.

    Handles:
    - Configuration validation on startup
    - ClaudeConnector initialization with connection pooling
    - Resource cleanup on shutdown
    """
    settings = get_settings()

    # Initialize connector once for the app lifetime
    connector: JobExtractor
    if settings.mock_llm:
        connector = MockClaudeConnector()
    else:
        connector = ClaudeConnector(settings=settings)
    await connector.initialize()
    app.state.claude_connector = connector

    yield

    # Cleanup on shutdown
    await connector.cleanup()


def business_exception_handler(_request: Request, exc: BusinessError) -> JSONResponse:
    """Handle application-specific business exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "error_code": exc.error_code},
    )


def general_exception_handler(_request: Request, _exc: Exception) -> JSONResponse:
    """Catch-all handler for unexpected exceptions."""
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


def create_app() -> FastAPI:
    """
    Application factory using composition pattern.

    Creates and configures the FastAPI application with:
    - Exception handlers
    - API routers
    - Health check endpoints
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Exception handlers
    # Ignore arg-type: FastAPI's add_exception_handler typing is overly strict
    # and doesn't account for exception subclasses being valid handlers
    app.add_exception_handler(BusinessError, business_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, general_exception_handler)

    # Register API routers
    # (Lazy router import inside the function to avoid circular imports and
    # reduce startup time if the factory isn't called)
    from job_posting_extractor.api.routers import extraction_router

    app.include_router(extraction_router, prefix="/api/v1")

    # Health check endpoints
    @app.get("/health")
    async def health_check() -> dict[str, str]:
        return {"status": "healthy", "version": __version__}

    return app


def start_api() -> None:
    """Entry point for starting the API server."""
    import uvicorn

    settings = get_settings()

    uvicorn.run(
        "job_posting_extractor.api.service:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level,
    )
