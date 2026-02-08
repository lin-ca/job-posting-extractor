"""Tests for ExtractionService."""

from unittest.mock import AsyncMock

import pytest

from job_posting_extractor.exceptions import ExtractionError
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


def _create_job(**kwargs: object) -> JobPosting:
    """Helper to create job postings with specific fields."""
    defaults = {"job_title": "Developer", "company": "TestCo"}
    return JobPosting(**{**defaults, **kwargs})  # type: ignore[arg-type]


def _create_result(job: JobPosting) -> RawExtractionResult:
    """Helper to create extraction results."""
    return RawExtractionResult(
        job=job,
        raw_response="{}",
        model="test-model",
        usage=UsageInfo(input_tokens=10, output_tokens=20),
    )


class TestExtractionService:
    """Tests for ExtractionService."""

    async def test_extract_job_returns_response(
        self,
        mock_job_extractor: AsyncMock,
        sample_job_posting: JobPosting,
    ) -> None:
        service = ExtractionService(connector=mock_job_extractor)
        result = await service.extract_job("Sample job posting text")

        assert result.job == sample_job_posting
        assert result.confidence in ["high", "medium", "low"]
        assert result.model == TEST_MODEL
        mock_job_extractor.extract_job_posting.assert_called_once_with(
            "Sample job posting text"
        )

    async def test_cleanup_delegates_to_connector(
        self, mock_job_extractor: AsyncMock
    ) -> None:
        service = ExtractionService(connector=mock_job_extractor)
        await service.cleanup()
        mock_job_extractor.cleanup.assert_called_once()


class TestConfidenceCalculation:
    """Tests for confidence score calculation."""

    @pytest.mark.parametrize(
        ("job_kwargs", "expected_confidence"),
        [
            # High confidence: 6+ optional fields
            pytest.param(
                {
                    "location": "Berlin",
                    "work_location": WorkLocation.REMOTE,
                    "employment_type": EmploymentType.FULL_TIME,
                    "experience_level": ExperienceLevel.SENIOR,
                    "salary": SalaryRange(min=50000, max=80000),
                    "requirements": ["Python"],
                    "responsibilities": ["Code"],
                    "benefits": ["Health insurance"],
                },
                "high",
                id="many_fields",
            ),
            # High confidence boundary: exactly 6 fields
            pytest.param(
                {
                    "location": "Berlin",
                    "work_location": WorkLocation.REMOTE,
                    "employment_type": EmploymentType.CONTRACT,
                    "experience_level": ExperienceLevel.MID,
                    "salary": SalaryRange(min=40000),
                    "requirements": ["Python"],
                },
                "high",
                id="boundary_at_six",
            ),
            # Medium confidence: 3-5 optional fields
            pytest.param(
                {
                    "location": "Berlin",
                    "work_location": WorkLocation.HYBRID,
                    "employment_type": EmploymentType.FULL_TIME,
                },
                "medium",
                id="some_fields",
            ),
            # Medium confidence boundary: exactly 3 fields
            pytest.param(
                {
                    "location": "Berlin",
                    "work_location": WorkLocation.ON_SITE,
                    "requirements": ["JavaScript"],
                },
                "medium",
                id="boundary_at_three",
            ),
            # Low confidence: 0-2 optional fields
            pytest.param(
                {"location": "Berlin"},
                "low",
                id="few_fields",
            ),
            # Low confidence: no optional fields
            pytest.param(
                {},
                "low",
                id="no_optional_fields",
            ),
            # Low confidence: empty lists don't count
            pytest.param(
                {"requirements": [], "responsibilities": [], "benefits": []},
                "low",
                id="empty_lists_not_counted",
            ),
        ],
    )
    async def test_confidence_levels(
        self,
        mock_job_extractor: AsyncMock,
        job_kwargs: dict,
        expected_confidence: str,
    ) -> None:
        job = _create_job(**job_kwargs)
        mock_job_extractor.extract_job_posting.return_value = _create_result(job)

        service = ExtractionService(connector=mock_job_extractor)
        result = await service.extract_job("job text")

        assert result.confidence == expected_confidence


class TestExtractionServiceErrorPaths:
    """Tests for error propagation from connector to service."""

    async def test_extraction_error_propagates(
        self, mock_job_extractor: AsyncMock
    ) -> None:
        mock_job_extractor.extract_job_posting.side_effect = ExtractionError(
            "Failed to extract job data"
        )
        service = ExtractionService(connector=mock_job_extractor)

        with pytest.raises(ExtractionError, match="Failed to extract job data"):
            await service.extract_job("Some job posting")

    async def test_unexpected_error_propagates(
        self, mock_job_extractor: AsyncMock
    ) -> None:
        mock_job_extractor.extract_job_posting.side_effect = RuntimeError(
            "Connection lost"
        )
        service = ExtractionService(connector=mock_job_extractor)

        with pytest.raises(RuntimeError, match="Connection lost"):
            await service.extract_job("Some job posting")

    async def test_cleanup_error_propagates(
        self, mock_job_extractor: AsyncMock
    ) -> None:
        mock_job_extractor.cleanup.side_effect = RuntimeError("Cleanup failed")
        service = ExtractionService(connector=mock_job_extractor)

        with pytest.raises(RuntimeError, match="Cleanup failed"):
            await service.cleanup()


class TestExtractionServiceWithMockConnector:
    """Integration tests using MockClaudeConnector."""

    async def test_extract_with_mock_connector(
        self, extraction_service: ExtractionService
    ) -> None:
        result = await extraction_service.extract_job("Any job posting text")

        assert result.job.job_title == "Senior Python Developer"
        assert result.job.company == "TechCorp"
        assert result.confidence == "high"
        assert result.model == "mock-model"
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 200
