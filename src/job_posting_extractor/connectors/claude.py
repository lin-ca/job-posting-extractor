"""Claude API connector for interacting with Anthropic's Claude models."""

from typing import Any, Never, Self

import anthropic
import httpx
from anthropic.types import TextBlock, ToolParam, ToolUseBlock
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from job_posting_extractor.config import Settings
from job_posting_extractor.connectors.shared import (
    EXTRACTION_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    JOB_EXTRACTION_PROPERTIES,
    JOB_EXTRACTION_REQUIRED_FIELDS,
    RETRYABLE_STATUS_CODES,
    validate_message,
)
from job_posting_extractor.exceptions import ExtractionError
from job_posting_extractor.models import (
    ClaudeResponse,
    JobPosting,
    RawExtractionResult,
    UsageInfo,
)

JOB_EXTRACTION_TOOL: ToolParam = {
    "name": "extract_job_posting",
    "description": "Extract structured job posting information from text",
    "input_schema": {
        "type": "object",
        "properties": JOB_EXTRACTION_PROPERTIES,
        "required": JOB_EXTRACTION_REQUIRED_FIELDS,
    },
}


def _is_retryable_error(exc: BaseException) -> bool:
    """Determine if an API error is retryable."""
    # Network and rate limit errors are always retryable
    if isinstance(exc, (anthropic.RateLimitError, anthropic.APIConnectionError)):
        return True

    # Server errors (5xx) and rate limits (429) are retryable
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code in RETRYABLE_STATUS_CODES

    return False


class ClaudeConnector:
    """Connector for Claude API interactions."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = anthropic.AsyncAnthropic(
            api_key=settings.api_key.get_secret_value(),
            timeout=httpx.Timeout(settings.api_timeout, connect=10.0),
        )

    async def initialize(self) -> None:
        """Initialize connector resources. No-op as client is created in __init__."""

    async def cleanup(self) -> None:
        """Cleanup client resources."""
        await self.client.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.cleanup()

    async def health_check(self) -> dict[str, Any]:
        """Check Claude API connectivity."""
        try:
            response = await self.send_message(
                "Say 'ok' and nothing else.", max_tokens=10
            )
            return {"status": "healthy", "model": response.model}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    @retry(
        retry=retry_if_exception(_is_retryable_error),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def send_message(
        self, message: str, max_tokens: int | None = None
    ) -> ClaudeResponse:
        """Send a message to Claude and get a response."""
        self._validate_message(message)

        try:
            response = await self.client.messages.create(
                model=self.settings.claude_model,
                max_tokens=max_tokens or self.settings.max_tokens,
                messages=[{"role": "user", "content": message}],
            )

            if not response.content:
                raise ExtractionError("Claude returned empty response")

            content_block = response.content[0]
            if not isinstance(content_block, TextBlock):
                raise ExtractionError(
                    f"Unexpected response type: {type(content_block).__name__}"
                )

            return ClaudeResponse(
                response=content_block.text,
                model=response.model,
                usage=UsageInfo(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                ),
            )

        except anthropic.APIError as e:
            self._handle_api_error(e)

    @staticmethod
    def _validate_message(message: str, max_chars: int = 50_000) -> None:
        """Validate message before API call."""
        validate_message(message, max_chars)

    def _handle_api_error(self, error: anthropic.APIError) -> Never:
        """Re-raise retryable errors; wrap others in ExtractionError."""
        if _is_retryable_error(error):
            raise error
        raise ExtractionError(f"Claude API error: {error}") from error

    @retry(
        retry=retry_if_exception(_is_retryable_error),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def extract_job_posting(self, job_text: str) -> RawExtractionResult:
        """
        Extract structured job posting data from unstructured text.

        Uses Claude's tool_use feature for guaranteed structured output.

        Args:
            job_text: Raw job posting text to parse.

        Returns:
            RawExtractionResult with extracted job data and API metadata.

        Raises:
            ExtractionError: If extraction or parsing fails.
        """
        self._validate_message(job_text)

        try:
            response = await self.client.messages.create(
                model=self.settings.claude_model,
                max_tokens=self.settings.max_tokens,
                system=EXTRACTION_SYSTEM_PROMPT,
                tools=[JOB_EXTRACTION_TOOL],
                tool_choice={"type": "tool", "name": "extract_job_posting"},
                messages=[
                    {
                        "role": "user",
                        "content": f"{EXTRACTION_PROMPT}{job_text}",
                    }
                ],
            )

            # Get ToolUseBlock in case there are more Blocks (e.g. TextBlock, ThinkingBlock)
            tool_use_block = None
            for block in response.content:
                if isinstance(block, ToolUseBlock):
                    tool_use_block = block
                    break

            if tool_use_block is None:
                raise ExtractionError("Claude did not return structured output")

            job_data = tool_use_block.input
            if not isinstance(job_data, dict):
                raise ExtractionError(
                    f"Unexpected tool input type: {type(job_data).__name__}"
                )

            job = JobPosting(**job_data)

            return RawExtractionResult(
                job=job,
                raw_response=str(job_data),
                model=response.model,
                usage=UsageInfo(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                ),
            )

        except anthropic.APIError as e:
            self._handle_api_error(e)
        except ValidationError as e:
            raise ExtractionError(f"Invalid job data structure: {e}") from e
        except ExtractionError:
            raise
        except Exception as e:
            raise ExtractionError(f"Error extracting job posting: {e}") from e
