"""Mock Claude connector for testing and local development without API calls."""

from typing import Any

from job_posting_extractor.models import (
    EmploymentType,
    ExperienceLevel,
    JobPosting,
    RawExtractionResult,
    SalaryRange,
    UsageInfo,
    WorkLocation,
)

MOCK_JOB_POSTING = JobPosting(
    job_title="Senior Python Developer",
    company="TechCorp",
    location="Berlin, Germany",
    work_location=WorkLocation.HYBRID,
    employment_type=EmploymentType.FULL_TIME,
    experience_level=ExperienceLevel.SENIOR,
    salary=SalaryRange(min=70000, max=90000, currency="EUR"),
    requirements=[
        "5+ years Python experience",
        "Experience with FastAPI or Django",
        "Strong understanding of REST APIs",
        "Knowledge of PostgreSQL",
    ],
    nice_to_have=[
        "Experience with Docker/Kubernetes",
        "Cloud platform experience (AWS/GCP)",
    ],
    responsibilities=[],
    benefits=[
        "Competitive salary (70,000 - 90,000)",
        "30 days vacation",
        "Remote work flexibility",
        "Learning budget",
    ],
    application_url="https://techcorp.com/careers/python-dev",
    application_deadline=None,
    posted_date=None,
)

MOCK_RAW_RESPONSE = """{
  "job_title": "Senior Python Developer",
  "company": "TechCorp",
  "location": "Berlin, Germany",
  "work_location": "hybrid",
  "employment_type": "full_time",
  "experience_level": "senior",
  "salary": {"min": 70000, "max": 90000, "currency": "EUR"},
  "requirements": [
    "5+ years Python experience",
    "Experience with FastAPI or Django",
    "Strong understanding of REST APIs",
    "Knowledge of PostgreSQL"
  ],
  "nice_to_have": [
    "Experience with Docker/Kubernetes",
    "Cloud platform experience (AWS/GCP)"
  ],
  "responsibilities": [],
  "benefits": [
    "Competitive salary (70,000 - 90,000)",
    "30 days vacation",
    "Remote work flexibility",
    "Learning budget"
  ],
  "application_url": "https://techcorp.com/careers/python-dev",
  "application_deadline": null,
  "posted_date": null
}"""


class MockClaudeConnector:
    """Mock connector for testing and local development without API calls."""

    async def initialize(self) -> None:
        """No-op initialization for mock connector."""

    async def cleanup(self) -> None:
        """No-op cleanup for mock connector."""

    async def health_check(self) -> dict[str, Any]:
        """Return mock health status."""
        return {"status": "healthy", "model": "mock"}

    async def extract_job_posting(self, _job_text: str) -> RawExtractionResult:
        """Return mock job posting data."""
        return RawExtractionResult(
            job=MOCK_JOB_POSTING,
            raw_response=MOCK_RAW_RESPONSE,
            model="mock-model",
            usage=UsageInfo(input_tokens=100, output_tokens=200),
        )
