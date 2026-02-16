"""Integration tests for API endpoints."""

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from job_posting_extractor import __version__
from job_posting_extractor.api.dependencies import get_extraction_service
from job_posting_extractor.exceptions import ExtractionError
from job_posting_extractor.models import (
    JobExtractionResponse,
    JobPosting,
    UsageInfo,
)
from job_posting_extractor.services.extraction import ExtractionService


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check_returns_healthy(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == __version__


class TestExtractJobEndpoint:
    """Tests for POST /api/v1/extract/job endpoint."""

    def test_extract_job_success(
        self, client: TestClient, sample_job_text: str
    ) -> None:
        response = client.post("/api/v1/extract/job", json={"text": sample_job_text})

        assert response.status_code == 200
        data = response.json()
        assert "job" in data
        assert "confidence" in data
        assert data["job"]["job_title"] == "Senior Python Developer"
        assert data["job"]["company"] == "TechCorp"

    def test_extract_job_returns_confidence(
        self, client: TestClient, sample_job_text: str
    ) -> None:
        response = client.post("/api/v1/extract/job", json={"text": sample_job_text})

        data = response.json()
        assert data["confidence"] in ["high", "medium", "low"]

    def test_extract_job_returns_metadata(
        self, client: TestClient, sample_job_text: str
    ) -> None:
        response = client.post("/api/v1/extract/job", json={"text": sample_job_text})

        data = response.json()
        assert "model" in data
        assert "usage" in data
        assert "raw_response" in data
        assert data["usage"]["input_tokens"] > 0
        assert data["usage"]["output_tokens"] > 0

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param({"text": ""}, id="empty_text"),
            pytest.param({"text": "   \n\t   "}, id="whitespace_text"),
            pytest.param({}, id="missing_text"),
        ],
    )
    def test_extract_job_invalid_requests(
        self,
        client: TestClient,
        payload: dict,
    ) -> None:
        response = client.post("/api/v1/extract/job", json=payload)
        assert response.status_code == 422

    def test_extract_job_invalid_json(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/extract/job",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422


class TestExceptionHandlers:
    """Tests for custom exception handlers."""

    @pytest.fixture
    def app_with_failing_service(self, app: FastAPI) -> FastAPI:
        """App with a service that raises ExtractionError."""
        mock_service = AsyncMock(spec=ExtractionService)
        mock_service.extract_job.side_effect = ExtractionError(
            "Failed to extract job data"
        )

        def override_service() -> ExtractionService:
            return mock_service  # type: ignore[return-value]

        app.dependency_overrides[get_extraction_service] = override_service
        return app

    def test_extraction_error_returns_422(
        self, app_with_failing_service: FastAPI
    ) -> None:
        with TestClient(app_with_failing_service) as client:
            response = client.post(
                "/api/v1/extract/job", json={"text": "Some job posting"}
            )

        assert response.status_code == 422
        data = response.json()
        assert data["error_code"] == "EXTRACTION_ERROR"
        assert data["detail"] == "Failed to extract job data"


class TestDependencyOverrides:
    """Tests demonstrating dependency injection for testing."""

    @pytest.fixture
    def mock_extraction_response(
        self, minimal_job_posting: JobPosting, sample_usage_info: UsageInfo
    ) -> JobExtractionResponse:
        """Custom extraction response for testing."""
        return JobExtractionResponse(
            job=minimal_job_posting,
            confidence="low",
            raw_response='{"test": true}',
            model="test-model",
            usage=sample_usage_info,
        )

    def test_override_extraction_service(
        self,
        app: FastAPI,
        mock_extraction_response: JobExtractionResponse,
    ) -> None:
        """Demonstrate overriding the extraction service for isolated testing."""
        mock_service = AsyncMock(spec=ExtractionService)
        mock_service.extract_job.return_value = mock_extraction_response

        def override_service() -> ExtractionService:
            return mock_service  # type: ignore[return-value]

        app.dependency_overrides[get_extraction_service] = override_service

        client = TestClient(app)
        response = client.post("/api/v1/extract/job", json={"text": "Test posting"})

        assert response.status_code == 200
        data = response.json()
        assert data["job"]["job_title"] == "Developer"
        assert data["job"]["company"] == "StartupCo"
        assert data["confidence"] == "low"
        assert data["model"] == "test-model"

        mock_service.extract_job.assert_called_once_with("Test posting")


class TestResponseStructure:
    """Tests verifying the complete response structure."""

    def test_full_response_structure(
        self, client: TestClient, sample_job_text: str
    ) -> None:
        response = client.post("/api/v1/extract/job", json={"text": sample_job_text})
        data = response.json()

        # Top-level fields
        assert set(data.keys()) == {
            "job",
            "confidence",
            "raw_response",
            "model",
            "usage",
        }

        # Job fields
        job = data["job"]
        expected_job_fields = {
            "job_title",
            "company",
            "location",
            "work_location",
            "employment_type",
            "experience_level",
            "salary",
            "requirements",
            "nice_to_have",
            "responsibilities",
            "benefits",
            "application_url",
            "application_deadline",
            "posted_date",
        }
        assert set(job.keys()) == expected_job_fields

        # Usage fields
        usage = data["usage"]
        assert set(usage.keys()) == {"input_tokens", "output_tokens"}


class TestAppFactory:
    """Tests for application factory and configuration."""

    def test_app_includes_extraction_router(self, app: Any) -> None:
        routes = [route.path for route in app.routes]
        assert "/api/v1/extract/job" in routes

    def test_app_includes_health_endpoint(self, app: Any) -> None:
        routes = [route.path for route in app.routes]
        assert "/health" in routes

    def test_app_has_docs_enabled(self, app: Any) -> None:
        routes = [route.path for route in app.routes]
        assert "/docs" in routes
        assert "/redoc" in routes

    def test_app_title_from_settings(self, app: Any) -> None:
        assert app.title == "Job Posting Extractor"
