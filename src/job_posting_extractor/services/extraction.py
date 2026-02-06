"""Extraction service for orchestrating job posting extraction.

This service layer handles business logic for job extraction, including:
- Confidence calculation based on extraction completeness
- TODO: Response caching to avoid redundant API calls
- TODO: Database persistence for extracted jobs
- TODO: Batch extraction support
- TODO: Extraction analytics and metrics
"""

from job_posting_extractor.connectors.base import JobExtractor
from job_posting_extractor.models import Confidence, JobExtractionResponse, JobPosting


class ExtractionService:
    """Service for orchestrating extraction operations.

    This service wraps the Claude connector and applies business logic
    to raw extraction results.
    """

    def __init__(self, connector: JobExtractor) -> None:
        self._connector = connector

    async def cleanup(self) -> None:
        """Cleanup service resources."""
        await self._connector.cleanup()

    async def extract_job(self, job_text: str) -> JobExtractionResponse:
        """Extract structured job posting data from unstructured text.

        Args:
            job_text: Raw job posting text to parse.

        Returns:
            JobExtractionResponse with extracted data and confidence score.
        """
        result = await self._connector.extract_job_posting(job_text)
        confidence = self._calculate_confidence(result.job)

        response = JobExtractionResponse(
            job=result.job,
            confidence=confidence,
            raw_response=result.raw_response,
            model=result.model,
            usage=result.usage,
        )

        return response

    def _calculate_confidence(self, job: JobPosting) -> Confidence:
        """Calculate confidence based on optional field completeness.

        Since job_title and company are required, we measure extraction
        quality by how many optional fields were successfully extracted:
        - high: 6+ optional fields present
        - medium: 3-5 optional fields present
        - low: 0-2 optional fields present
        """
        optional_fields_present = sum(
            [
                job.location is not None,
                job.work_location is not None,
                job.employment_type is not None,
                job.experience_level is not None,
                job.salary is not None,
                len(job.requirements) > 0,
                len(job.nice_to_have) > 0,
                len(job.responsibilities) > 0,
                len(job.benefits) > 0,
            ]
        )

        if optional_fields_present >= 6:
            return "high"
        elif optional_fields_present >= 3:
            return "medium"
        return "low"
