"""Shared test fixtures for job_posting_extractor tests."""

from datetime import date
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from job_posting_extractor.api.service import create_app
from job_posting_extractor.connectors.mock_claude import MockClaudeConnector
from job_posting_extractor.models import (
    EmploymentType,
    ExperienceLevel,
    JobPosting,
    RawExtractionResult,
    SalaryRange,
    UsageInfo,
    WorkLocation,
)
from job_posting_extractor.services.extraction import ExtractionService
from tests import TEST_MODEL


@pytest.fixture
def sample_job_posting() -> JobPosting:
    """A fully populated job posting for testing."""
    return JobPosting(
        job_title="Senior Python Developer",
        company="TechCorp",
        location="Berlin, Germany",
        work_location=WorkLocation.HYBRID,
        employment_type=EmploymentType.FULL_TIME,
        experience_level=ExperienceLevel.SENIOR,
        salary=SalaryRange(min=70000, max=90000, currency="EUR"),
        requirements=[
            "5+ years Python experience",
            "Experience with FastAPI",
        ],
        nice_to_have=["Docker experience"],
        responsibilities=["Design APIs", "Code reviews"],
        benefits=["30 days vacation", "Remote flexibility"],
        application_url="https://example.com/apply",
        application_deadline=date(2025, 12, 31),
        posted_date=date(2025, 1, 15),
    )


@pytest.fixture
def minimal_job_posting() -> JobPosting:
    """A job posting with only required fields."""
    return JobPosting(
        job_title="Developer",
        company="StartupCo",
    )


@pytest.fixture
def sample_usage_info() -> UsageInfo:
    """Sample token usage info."""
    return UsageInfo(input_tokens=150, output_tokens=300)


@pytest.fixture
def sample_raw_extraction_result(
    sample_job_posting: JobPosting, sample_usage_info: UsageInfo
) -> RawExtractionResult:
    """Sample raw extraction result."""
    return RawExtractionResult(
        job=sample_job_posting,
        raw_response='{"job_title": "Senior Python Developer"}',
        model=TEST_MODEL,
        usage=sample_usage_info,
    )


@pytest.fixture
def mock_connector() -> MockClaudeConnector:
    """Mock Claude connector for testing."""
    return MockClaudeConnector()


@pytest.fixture
def extraction_service(mock_connector: MockClaudeConnector) -> ExtractionService:
    """Extraction service with mock connector."""
    return ExtractionService(connector=mock_connector)


@pytest.fixture
def mock_job_extractor(
    sample_raw_extraction_result: RawExtractionResult,
) -> AsyncMock:
    """A mock JobExtractor for unit testing."""
    mock = AsyncMock()
    mock.extract_job_posting.return_value = sample_raw_extraction_result
    mock.initialize.return_value = None
    mock.cleanup.return_value = None
    mock.health_check.return_value = {"status": "healthy", "model": "mock"}
    return mock


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Create test application with mock LLM enabled."""
    monkeypatch.setenv("MOCK_LLM", "true")
    from job_posting_extractor.config import get_settings

    get_settings.cache_clear()
    yield create_app()
    get_settings.cache_clear()


@pytest.fixture
def client(app: Any) -> TestClient:
    """Test client for API integration tests with lifespan support."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_job_text() -> str:
    """Sample job posting text for extraction tests."""
    return """
    Senior Python Developer - TechCorp

    Location: Berlin, Germany (Hybrid)
    Type: Full-time
    Experience: Senior level

    About the role:
    We're looking for an experienced Python developer.

    Requirements:
    - 5+ years Python experience
    - Experience with FastAPI

    Nice to have:
    - Docker experience

    Responsibilities:
    - Design and implement APIs
    - Conduct code reviews

    Benefits:
    - 30 days vacation
    - Remote work flexibility

    Salary: €70,000 - €90,000

    Apply at: https://techcorp.com/careers
    """
