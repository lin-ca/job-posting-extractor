"""OpenAI-compatible connector for local LLM servers (LM Studio, Ollama, vLLM, etc.)."""

import json
import logging
from typing import Any, Never, Self

import httpx
from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from job_posting_extractor.config import Settings
from job_posting_extractor.connectors.shared import (
    ALL_JOB_FIELDS,
    EXTRACTION_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    JOB_EXTRACTION_PROPERTIES,
    JOB_EXTRACTION_REQUIRED_FIELDS,
    RETRYABLE_STATUS_CODES,
    validate_message,
)
from job_posting_extractor.exceptions import ExtractionError
from job_posting_extractor.models import (
    JobPosting,
    RawExtractionResult,
    UsageInfo,
)

logger = logging.getLogger(__name__)

JOB_EXTRACTION_FUNCTION = {
    "type": "function",
    "function": {
        "name": "extract_job_posting",
        "description": "Extract structured job posting information from text",
        "parameters": {
            "type": "object",
            "properties": JOB_EXTRACTION_PROPERTIES,
            "required": JOB_EXTRACTION_REQUIRED_FIELDS,
        },
    },
}

def _to_strict_property(prop: dict[str, Any]) -> dict[str, Any]:
    """Convert a JSON Schema property to strict-mode compatible format.

    LM Studio / OpenAI strict mode requires "type" to be a string, not an array.
    Convert {"type": ["string", "null"]} to {"anyOf": [{"type": "string"}, {"type": "null"}]}.
    """
    prop = dict(prop)  # shallow copy
    if isinstance(prop.get("type"), list):
        types = prop.pop("type")
        description = prop.pop("description", None)
        enum = prop.pop("enum", None)
        # Build anyOf from the type list
        branches: list[dict[str, Any]] = []
        for t in types:
            if t == "null":
                branches.append({"type": "null"})
            else:
                branch: dict[str, Any] = {"type": t}
                if enum is not None:
                    branch["enum"] = [v for v in enum if v is not None]
                # For object types, include nested properties
                if t == "object" and "properties" in prop:
                    branch["properties"] = {
                        k: _to_strict_property(v)
                        for k, v in prop.pop("properties").items()
                    }
                    branch["required"] = list(branch["properties"].keys())
                    branch["additionalProperties"] = False
                branches.append(branch)
        result: dict[str, Any] = {"anyOf": branches}
        if description:
            result["description"] = description
        return result
    return prop


def _build_strict_schema() -> dict[str, Any]:
    """Build a strict-mode JSON schema from the shared properties."""
    strict_props = {
        k: _to_strict_property(v) for k, v in JOB_EXTRACTION_PROPERTIES.items()
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "job_posting",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": strict_props,
                # Strict mode requires all properties in "required".
                # Optional fields use anyOf with null type instead.
                "required": ALL_JOB_FIELDS,
                "additionalProperties": False,
            },
        },
    }


JOB_EXTRACTION_SCHEMA = _build_strict_schema()


def _is_retryable_error(exc: BaseException) -> bool:
    """Determine if an API error is retryable."""
    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in RETRYABLE_STATUS_CODES
    return False


class OpenAICompatConnector:
    """Connector for OpenAI-compatible APIs (LM Studio, Ollama, vLLM, etc.)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=settings.openai_base_url,
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
        """Check API connectivity by listing models."""
        try:
            models = await self.client.models.list()
            return {
                "status": "healthy",
                "model": self.settings.openai_model,
                "available_models": [m.id for m in models.data[:5]],
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def _handle_api_error(self, error: APIStatusError) -> Never:
        """Re-raise retryable errors; wrap others in ExtractionError."""
        if _is_retryable_error(error):
            raise error
        raise ExtractionError(f"OpenAI-compatible API error: {error}") from error

    @retry(
        retry=retry_if_exception(_is_retryable_error),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def extract_job_posting(self, job_text: str) -> RawExtractionResult:
        """
        Extract structured job posting data from unstructured text.

        Tries function calling first. If the model doesn't support it,
        falls back to structured output via constrained decoding (response_format).
        """
        validate_message(job_text)

        try:
            return await self._extract_with_function_calling(job_text)
        except ExtractionError:
            raise
        except APIConnectionError:
            raise  # Server unreachable -- no point falling back
        except (APIStatusError, ValueError, NotImplementedError) as e:
            logger.warning(
                "Function calling failed (%s: %s), falling back to structured output",
                type(e).__name__,
                e,
            )
            return await self._extract_with_structured_output(job_text)

    async def _extract_with_function_calling(
        self, job_text: str
    ) -> RawExtractionResult:
        """Extract using OpenAI function calling.

        Raises raw APIStatusError / APIConnectionError on failure so that
        the caller can fall back to structured output.
        """
        response = await self.client.chat.completions.create(
            model=self.settings.openai_model,
            max_tokens=self.settings.max_tokens,
            tools=[JOB_EXTRACTION_FUNCTION],  # type: ignore[list-item]
            tool_choice="required",
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"{EXTRACTION_PROMPT}{job_text}",
                },
            ],
        )

        choice = response.choices[0]
        tool_calls = choice.message.tool_calls

        if not tool_calls:
            raise ValueError("Model did not return a function call")

        arguments = tool_calls[0].function.arguments
        try:
            job_data = json.loads(arguments)
        except json.JSONDecodeError as e:
            raise ExtractionError(
                f"Model returned invalid JSON in function call: {e}"
            ) from e
        return self._build_result(job_data, response)

    async def _extract_with_structured_output(
        self, job_text: str
    ) -> RawExtractionResult:
        """Fallback: extract using constrained decoding via response_format."""
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.openai_model,
                max_tokens=self.settings.max_tokens,
                temperature=0,
                response_format=JOB_EXTRACTION_SCHEMA,  # type: ignore[arg-type]
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"{EXTRACTION_PROMPT}{job_text}",
                    },
                ],
            )

            content = response.choices[0].message.content
            if not content:
                raise ExtractionError("Model returned empty response")

            job_data = json.loads(content)
            return self._build_result(job_data, response)

        except APIStatusError as e:
            self._handle_api_error(e)
        except json.JSONDecodeError as e:
            raise ExtractionError(f"Model returned invalid JSON: {e}") from e

    def _build_result(
        self, job_data: dict[str, Any], response: Any
    ) -> RawExtractionResult:
        """Build a RawExtractionResult from parsed job data and API response."""
        try:
            job = JobPosting(**job_data)
        except ValidationError as e:
            raise ExtractionError(f"Invalid job data structure: {e}") from e

        usage = response.usage
        return RawExtractionResult(
            job=job,
            raw_response=str(job_data),
            model=response.model,
            usage=UsageInfo(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            ),
        )
