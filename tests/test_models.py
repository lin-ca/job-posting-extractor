"""Tests for Pydantic models."""

from datetime import date

import pytest
from pydantic import HttpUrl, ValidationError

from job_posting_extractor.models import (
    ClaudeResponse,
    EmploymentType,
    ExperienceLevel,
    JobExtractionRequest,
    JobExtractionResponse,
    JobPosting,
    RawExtractionResult,
    SalaryRange,
    UsageInfo,
    WorkLocation,
)
from tests import TEST_MODEL


class TestUsageInfo:
    """Tests for UsageInfo model."""

    def test_valid_usage_info(self) -> None:
        usage = UsageInfo(input_tokens=100, output_tokens=200)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 200


class TestClaudeResponse:
    """Tests for ClaudeResponse model."""

    def test_valid_claude_response(self) -> None:
        response = ClaudeResponse(
            response="Hello",
            model=TEST_MODEL,
            usage=UsageInfo(input_tokens=10, output_tokens=5),
        )
        assert response.response == "Hello"
        assert response.model == TEST_MODEL


class TestSalaryRange:
    """Tests for SalaryRange model."""

    def test_full_salary_range(self) -> None:
        salary = SalaryRange(min=50000, max=80000, currency="USD")
        assert salary.min == 50000
        assert salary.max == 80000
        assert salary.currency == "USD"

    def test_partial_salary_range(self) -> None:
        salary = SalaryRange(min=60000)
        assert salary.min == 60000
        assert salary.max is None


class TestJobPosting:
    """Tests for JobPosting model."""

    def test_minimal_job_posting(self) -> None:
        job = JobPosting(job_title="Developer", company="Some Comp")
        assert job.job_title == "Developer"
        assert job.company == "Some Comp"
        assert job.location is None
        assert job.requirements == []
        assert job.benefits == []

    def test_full_job_posting(self, sample_job_posting: JobPosting) -> None:
        assert sample_job_posting.job_title == "Senior Python Developer"
        assert sample_job_posting.company == "TechCorp"
        assert sample_job_posting.location == "Berlin, Germany"
        assert sample_job_posting.work_location == WorkLocation.HYBRID
        assert sample_job_posting.employment_type == EmploymentType.FULL_TIME
        assert sample_job_posting.experience_level == ExperienceLevel.SENIOR

    def test_job_posting_requires_title_and_company(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            JobPosting()  # type: ignore[call-arg]
        errors = exc_info.value.errors()
        assert len(errors) == 2
        assert {e["loc"][0] for e in errors} == {"job_title", "company"}

    def test_job_posting_with_valid_url(self) -> None:
        # Passing str where HttpUrl is expected to test Pydantic coercion
        job = JobPosting(
            job_title="Dev",
            company="Co",
            application_url="https://example.com/apply",  # type: ignore[arg-type]
        )
        assert isinstance(job.application_url, HttpUrl)
        assert str(job.application_url) == "https://example.com/apply"

    def test_job_posting_with_invalid_url(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            JobPosting(
                job_title="Dev",
                company="Co",
                application_url="not-a-url",  # type: ignore[arg-type]
            )
        assert "application_url" in str(exc_info.value)

    def test_job_posting_with_dates(self) -> None:
        job = JobPosting(
            job_title="Dev",
            company="Co",
            application_deadline=date(2025, 6, 30),
            posted_date=date(2025, 1, 1),
        )
        assert job.application_deadline == date(2025, 6, 30)
        assert job.posted_date == date(2025, 1, 1)

    def test_job_posting_enum_coercion(self) -> None:
        """Enums can be set from string values."""
        # Passing str where enums are expected to test Pydantic coercion
        job = JobPosting(
            job_title="Dev",
            company="Co",
            work_location="remote",  # type: ignore[arg-type]
            employment_type="contract",  # type: ignore[arg-type]
            experience_level="mid",  # type: ignore[arg-type]
        )
        assert job.work_location == WorkLocation.REMOTE
        assert job.employment_type == EmploymentType.CONTRACT
        assert job.experience_level == ExperienceLevel.MID


class TestJobExtractionRequest:
    """Tests for JobExtractionRequest model."""

    def test_valid_request(self) -> None:
        request = JobExtractionRequest(text="Software Engineer at Google")
        assert request.text == "Software Engineer at Google"

    def test_request_rejects_empty_text(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            JobExtractionRequest(text="")
        assert "String should have at least 1 character" in str(exc_info.value)

    def test_request_rejects_whitespace_only(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            JobExtractionRequest(text="   \n\t  ")
        assert "whitespace" in str(exc_info.value).lower()

    def test_request_accepts_text_with_whitespace(self) -> None:
        request = JobExtractionRequest(text="  Valid text  ")
        assert request.text == "  Valid text  "


class TestRawExtractionResult:
    """Tests for RawExtractionResult model."""

    def test_raw_extraction_result(
        self, sample_job_posting: JobPosting, sample_usage_info: UsageInfo
    ) -> None:
        result = RawExtractionResult(
            job=sample_job_posting,
            raw_response='{"key": "value"}',
            model=TEST_MODEL,
            usage=sample_usage_info,
        )
        assert result.job == sample_job_posting
        assert result.raw_response == '{"key": "value"}'
        assert result.model == TEST_MODEL


class TestJobExtractionResponse:
    """Tests for JobExtractionResponse model."""

    def test_job_extraction_response(
        self, sample_job_posting: JobPosting, sample_usage_info: UsageInfo
    ) -> None:
        response = JobExtractionResponse(
            job=sample_job_posting,
            confidence="high",
            raw_response='{"extracted": true}',
            model=TEST_MODEL,
            usage=sample_usage_info,
        )
        assert response.job == sample_job_posting
        assert response.confidence == "high"
        assert response.model == TEST_MODEL

    @pytest.mark.parametrize("confidence", ["high", "medium", "low"])
    def test_confidence_literal_values(
        self,
        confidence: str,
        minimal_job_posting: JobPosting,
        sample_usage_info: UsageInfo,
    ) -> None:
        response = JobExtractionResponse(
            job=minimal_job_posting,
            confidence=confidence,  # type: ignore[arg-type]
            raw_response="{}",
            model="test",
            usage=sample_usage_info,
        )
        assert response.confidence == confidence

    def test_invalid_confidence_rejected(
        self, minimal_job_posting: JobPosting, sample_usage_info: UsageInfo
    ) -> None:
        with pytest.raises(ValidationError):
            JobExtractionResponse(
                job=minimal_job_posting,
                confidence="invalid",  # type: ignore[arg-type]
                raw_response="{}",
                model="test",
                usage=sample_usage_info,
            )
