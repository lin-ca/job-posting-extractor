"""Extraction-related API endpoints."""

from fastapi import APIRouter

from job_posting_extractor.api.dependencies import ExtractionServiceDep
from job_posting_extractor.models import (
    JobExtractionRequest,
    JobExtractionResponse,
)

router = APIRouter(tags=["extraction"])


@router.post(
    "/extract/job",
    response_model=JobExtractionResponse,
    summary="Extract structured data from job posting text",
)
async def extract_job_handler(
    request: JobExtractionRequest,
    service: ExtractionServiceDep,
) -> JobExtractionResponse:
    """
    Extract structured job posting data from unstructured text.

    Analyzes job posting text and returns structured JSON with:
    - Job details (title, company, location, type)
    - Salary information
    - Requirements and nice-to-haves
    - Responsibilities
    - Benefits
    - Application info
    """
    return await service.extract_job(request.text)
