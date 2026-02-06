"""Pydantic models for request/response validation."""

from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator

type Confidence = Literal["high", "medium", "low"]


class UsageInfo(BaseModel):
    input_tokens: int
    output_tokens: int


class ClaudeResponse(BaseModel):
    response: str
    model: str
    usage: UsageInfo


class EmploymentType(StrEnum):
    """Employment type options."""

    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    TEMPORARY = "temporary"
    INTERNSHIP = "internship"
    FREELANCE = "freelance"


class WorkLocation(StrEnum):
    """Work location options."""

    REMOTE = "remote"
    HYBRID = "hybrid"
    ON_SITE = "on_site"


class ExperienceLevel(StrEnum):
    """Experience level options."""

    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"


class SalaryRange(BaseModel):
    """Salary information."""

    min: int | None = Field(default=None, description="Minimum salary")
    max: int | None = Field(default=None, description="Maximum salary")
    currency: str = Field(default="EUR", description="Currency code (ISO 4217)")


class JobPosting(BaseModel):
    """Structured job posting data."""

    job_title: str = Field(description="Job title/position")
    company: str = Field(description="Company name")
    location: str | None = Field(
        default=None, description="Location (city, state, country)"
    )
    work_location: WorkLocation | None = Field(
        default=None, description="Remote/hybrid/on-site"
    )
    employment_type: EmploymentType | None = Field(
        default=None, description="Employment type"
    )
    experience_level: ExperienceLevel | None = Field(
        default=None, description="Experience level required"
    )

    salary: SalaryRange | None = Field(
        default=None, description="Salary range if mentioned"
    )

    requirements: list[str] = Field(
        default_factory=list, description="Required skills/qualifications"
    )
    nice_to_have: list[str] = Field(
        default_factory=list, description="Preferred/nice-to-have skills"
    )
    responsibilities: list[str] = Field(
        default_factory=list, description="Key responsibilities"
    )
    benefits: list[str] = Field(default_factory=list, description="Benefits offered")

    application_url: HttpUrl | None = Field(
        default=None, description="Application URL if available"
    )
    application_deadline: date | None = Field(
        default=None, description="Application deadline if mentioned"
    )

    posted_date: date | None = Field(
        default=None, description="Job posting date if available"
    )


class RawExtractionResult(BaseModel):
    """Raw extraction result from Claude API (no business logic applied)."""

    job: JobPosting
    raw_response: str = Field(description="Raw Claude response for debugging")
    model: str = Field(description="Claude model used for extraction")
    usage: UsageInfo = Field(description="Token usage information")


class JobExtractionRequest(BaseModel):
    """Request to extract job posting data from text."""

    text: str = Field(
        min_length=1,
        description="Unstructured job posting text to parse",
        examples=[
            """Senior Python Developer - TechCorp

            Location: Berlin, Germany (Hybrid)
            Type: Full-time

            About the role:
            We're looking for an experienced Python developer to join our backend team.

            Requirements:
            - 5+ years Python experience
            - Experience with FastAPI or Django
            - Strong understanding of REST APIs
            - Knowledge of PostgreSQL

            Nice to have:
            - Experience with Docker/Kubernetes
            - Cloud platform experience (AWS/GCP)

            What we offer:
            - Competitive salary (€70,000 - €90,000)
            - 30 days vacation
            - Remote work flexibility
            - Learning budget

            Apply at: https://techcorp.com/careers/python-dev
            """
        ],
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_whitespace(cls, v: str) -> str:
        """Validate that text contains non-whitespace characters."""
        if not v.strip():
            raise ValueError("Text cannot be empty or contain only whitespace")
        return v


class JobExtractionResponse(BaseModel):
    """Response containing extracted job posting data."""

    job: JobPosting
    confidence: Confidence = Field(description="Confidence level: high, medium, or low")
    raw_response: str = Field(description="Raw Claude response for debugging")
    model: str = Field(description="Claude model used for extraction")
    usage: UsageInfo = Field(description="Token usage information")
