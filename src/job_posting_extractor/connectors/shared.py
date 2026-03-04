"""Shared constants and helpers for connectors."""

from typing import Any

from job_posting_extractor.exceptions import ExtractionError

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

EXTRACTION_SYSTEM_PROMPT = (
    "You are a structured data extraction tool. "
    "Extract job posting information and return it as structured data. "
    "Use null for any field not explicitly mentioned in the input. "
    "Do not guess or fabricate values — if information is not present, use null."
)

EXTRACTION_PROMPT = (
    "Extract the job posting information from the following text. "
    "Be thorough in extracting all mentioned skills, requirements, "
    "and benefits.\n\n"
)

JOB_EXTRACTION_PROPERTIES: dict[str, Any] = {
    "job_title": {"type": "string", "description": "Job title/position"},
    "company": {"type": "string", "description": "Company name"},
    "location": {
        "type": ["string", "null"],
        "description": "Location (city, state, country)",
    },
    "work_location": {
        "type": ["string", "null"],
        "enum": ["remote", "hybrid", "on_site", None],
        "description": "Remote/hybrid/on-site work arrangement",
    },
    "employment_type": {
        "type": ["string", "null"],
        "enum": [
            "full_time",
            "part_time",
            "contract",
            "temporary",
            "internship",
            "freelance",
            None,
        ],
        "description": "Type of employment",
    },
    "experience_level": {
        "type": ["string", "null"],
        "enum": ["entry", "mid", "senior", "lead", None],
        "description": "Required experience level",
    },
    "salary": {
        "type": ["object", "null"],
        "properties": {
            "min": {"type": ["integer", "null"]},
            "max": {"type": ["integer", "null"]},
            "currency": {"type": "string", "default": "EUR"},
        },
        "description": "Salary range information",
    },
    "requirements": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Required skills/qualifications",
    },
    "nice_to_have": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Preferred/nice-to-have skills",
    },
    "responsibilities": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Key job responsibilities",
    },
    "benefits": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Benefits offered",
    },
    "application_url": {
        "type": ["string", "null"],
        "description": "URL to apply",
    },
    "application_deadline": {
        "type": ["string", "null"],
        "description": "Deadline in YYYY-MM-DD format",
    },
    "posted_date": {
        "type": ["string", "null"],
        "description": "Posting date in YYYY-MM-DD format",
    },
}

JOB_EXTRACTION_REQUIRED_FIELDS = ["job_title", "company"]

ALL_JOB_FIELDS = list(JOB_EXTRACTION_PROPERTIES.keys())


def validate_message(message: str, max_chars: int = 50_000) -> None:
    """Validate message before API call.

    Cut off after 50000 characters to avoid using too many tokens.
    """
    if not message or not message.strip():
        raise ExtractionError("Message cannot be empty")

    if len(message) > max_chars:
        raise ExtractionError(
            f"Message too long ({len(message):,} chars). "
            f"Maximum supported length is {max_chars:,} characters."
        )
