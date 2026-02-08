"""Tests for ClaudeConnector."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest
from pydantic import SecretStr

from job_posting_extractor.config import Settings
from job_posting_extractor.connectors.claude import (
    ClaudeConnector,
    _is_retryable_error,
)
from job_posting_extractor.exceptions import ExtractionError
from tests import TEST_MODEL


@pytest.fixture
def mock_settings() -> Settings:
    """Settings with mock API key."""
    return Settings(
        anthropic_api_key=SecretStr("test-api-key"),
        claude_model=TEST_MODEL,
        max_tokens=1024,
        mock_llm=False,
    )


@pytest.fixture
def mock_anthropic_client() -> AsyncMock:
    """Mock Anthropic async client."""
    return AsyncMock(spec=anthropic.AsyncAnthropic)


@pytest.fixture
def connector_with_mock_client(
    mock_settings: Settings,
) -> tuple[ClaudeConnector, AsyncMock]:
    """ClaudeConnector with mocked Anthropic client."""
    mock_client = AsyncMock()
    with patch(
        "job_posting_extractor.connectors.claude.anthropic.AsyncAnthropic",
        return_value=mock_client,
    ):
        connector = ClaudeConnector(mock_settings)
        return connector, mock_client


class TestIsRetryableError:
    """Tests for _is_retryable_error helper function."""

    def test_rate_limit_error_is_retryable(self) -> None:
        error = anthropic.RateLimitError(
            message="rate limited",
            response=MagicMock(),
            body=None,
        )
        assert _is_retryable_error(error) is True

    def test_connection_error_is_retryable(self) -> None:
        error = anthropic.APIConnectionError(request=MagicMock())
        assert _is_retryable_error(error) is True

    @pytest.mark.parametrize("status_code", [429, 500, 502, 503, 504])
    def test_server_errors_are_retryable(self, status_code: int) -> None:
        error = anthropic.APIStatusError(
            message="error",
            response=MagicMock(status_code=status_code),
            body=None,
        )
        assert _is_retryable_error(error) is True

    @pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
    def test_client_errors_are_not_retryable(self, status_code: int) -> None:
        error = anthropic.APIStatusError(
            message="error",
            response=MagicMock(status_code=status_code),
            body=None,
        )
        assert _is_retryable_error(error) is False

    @pytest.mark.parametrize(
        "exception",
        [ValueError("test"), RuntimeError("test")],
        ids=["ValueError", "RuntimeError"],
    )
    def test_other_exceptions_not_retryable(self, exception: Exception) -> None:
        assert _is_retryable_error(exception) is False


class TestClaudeConnectorInit:
    """Tests for ClaudeConnector initialization."""

    def test_init_requires_api_key(self) -> None:
        settings = Settings(mock_llm=True)
        settings.anthropic_api_key = None

        with pytest.raises(AssertionError, match="API key required"):
            ClaudeConnector(settings)

    def test_init_with_valid_settings(self, mock_settings: Settings) -> None:
        with patch("job_posting_extractor.connectors.claude.anthropic.AsyncAnthropic"):
            connector = ClaudeConnector(mock_settings)
            assert connector.settings == mock_settings


class TestClaudeConnectorMessageValidation:
    """Tests for message validation."""

    @pytest.mark.parametrize(
        ("message", "error_match"),
        [
            ("", "cannot be empty"),
            ("   \n\t  ", "cannot be empty"),
            ("x" * 50_001, "too long"),
        ],
        ids=["empty", "whitespace", "too_long"],
    )
    def test_invalid_messages_raise_error(
        self,
        connector_with_mock_client: tuple[ClaudeConnector, AsyncMock],
        message: str,
        error_match: str,
    ) -> None:
        connector, _ = connector_with_mock_client
        with pytest.raises(ExtractionError, match=error_match):
            connector._validate_message(message)

    @pytest.mark.parametrize(
        "message",
        ["Valid message", "x" * 50_000],
        ids=["normal", "at_limit"],
    )
    def test_valid_messages_pass(
        self,
        connector_with_mock_client: tuple[ClaudeConnector, AsyncMock],
        message: str,
    ) -> None:
        connector, _ = connector_with_mock_client
        connector._validate_message(message)  # Should not raise


class TestClaudeConnectorSendMessage:
    """Tests for send_message method."""

    async def test_send_message_success(
        self, connector_with_mock_client: tuple[ClaudeConnector, AsyncMock]
    ) -> None:
        connector, mock_client = connector_with_mock_client

        from anthropic.types import TextBlock

        text_block = TextBlock(type="text", text="Hello, world!")

        mock_response = MagicMock()
        mock_response.content = [text_block]
        mock_response.model = TEST_MODEL
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5

        mock_client.messages.create.return_value = mock_response
        result = await connector.send_message("Hello")

        assert result.response == "Hello, world!"
        assert result.model == TEST_MODEL
        assert result.usage.input_tokens == 10

    async def test_send_message_empty_response(
        self, connector_with_mock_client: tuple[ClaudeConnector, AsyncMock]
    ) -> None:
        connector, mock_client = connector_with_mock_client

        mock_response = MagicMock()
        mock_response.content = []
        mock_client.messages.create.return_value = mock_response

        with pytest.raises(ExtractionError, match="empty response"):
            await connector.send_message("Hello")


class TestClaudeConnectorExtractJobPosting:
    """Tests for extract_job_posting method."""

    def _create_tool_use_response(self, job_data: dict[str, Any]) -> MagicMock:
        """Create a mock response with ToolUseBlock."""
        from anthropic.types import ToolUseBlock

        tool_block = ToolUseBlock(
            id="tool_123",
            type="tool_use",
            name="extract_job_posting",
            input=job_data,
        )
        mock_response = MagicMock()
        mock_response.content = [tool_block]
        mock_response.model = TEST_MODEL
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 200
        return mock_response

    async def test_extract_job_posting_success(
        self, connector_with_mock_client: tuple[ClaudeConnector, AsyncMock]
    ) -> None:
        connector, mock_client = connector_with_mock_client

        job_data = {
            "job_title": "Software Engineer",
            "company": "Acme Inc",
            "location": "San Francisco",
            "work_location": "remote",
            "employment_type": "full_time",
        }
        mock_client.messages.create.return_value = self._create_tool_use_response(
            job_data
        )

        result = await connector.extract_job_posting("Job posting text")

        assert result.job.job_title == "Software Engineer"
        assert result.job.company == "Acme Inc"
        assert result.job.location == "San Francisco"
        assert result.model == TEST_MODEL

    async def test_extract_job_posting_minimal_fields(
        self, connector_with_mock_client: tuple[ClaudeConnector, AsyncMock]
    ) -> None:
        connector, mock_client = connector_with_mock_client

        job_data = {"job_title": "Developer", "company": "StartupCo"}
        mock_client.messages.create.return_value = self._create_tool_use_response(
            job_data
        )

        result = await connector.extract_job_posting("Minimal job posting")

        assert result.job.job_title == "Developer"
        assert result.job.company == "StartupCo"
        assert result.job.location is None

    async def test_extract_job_posting_no_tool_use(
        self, connector_with_mock_client: tuple[ClaudeConnector, AsyncMock]
    ) -> None:
        connector, mock_client = connector_with_mock_client

        from anthropic.types import TextBlock

        text_block = TextBlock(type="text", text="I couldn't extract the data")
        mock_response = MagicMock()
        mock_response.content = [text_block]
        mock_client.messages.create.return_value = mock_response

        with pytest.raises(ExtractionError, match="did not return structured output"):
            await connector.extract_job_posting("Invalid job text")

    async def test_extract_job_posting_validation_error(
        self, connector_with_mock_client: tuple[ClaudeConnector, AsyncMock]
    ) -> None:
        connector, mock_client = connector_with_mock_client

        # Missing required fields
        job_data = {"location": "Berlin"}
        mock_client.messages.create.return_value = self._create_tool_use_response(
            job_data
        )

        with pytest.raises(ExtractionError, match="Invalid job data structure"):
            await connector.extract_job_posting("Bad job text")


class TestClaudeConnectorLifecycle:
    """Tests for connector lifecycle methods."""

    async def test_initialize_is_noop(
        self, connector_with_mock_client: tuple[ClaudeConnector, AsyncMock]
    ) -> None:
        connector, _ = connector_with_mock_client
        await connector.initialize()  # Should not raise

    async def test_cleanup_closes_client(
        self, connector_with_mock_client: tuple[ClaudeConnector, AsyncMock]
    ) -> None:
        connector, mock_client = connector_with_mock_client
        await connector.cleanup()
        mock_client.close.assert_called_once()

    async def test_async_context_manager(
        self, connector_with_mock_client: tuple[ClaudeConnector, AsyncMock]
    ) -> None:
        connector, mock_client = connector_with_mock_client
        async with connector:
            pass
        mock_client.close.assert_called_once()


class TestClaudeConnectorHealthCheck:
    """Tests for health check functionality."""

    async def test_health_check_healthy(
        self, connector_with_mock_client: tuple[ClaudeConnector, AsyncMock]
    ) -> None:
        connector, mock_client = connector_with_mock_client

        from anthropic.types import TextBlock

        text_block = TextBlock(type="text", text="ok")
        mock_response = MagicMock()
        mock_response.content = [text_block]
        mock_response.model = TEST_MODEL
        mock_response.usage.input_tokens = 5
        mock_response.usage.output_tokens = 1
        mock_client.messages.create.return_value = mock_response

        result = await connector.health_check()

        assert result["status"] == "healthy"
        assert result["model"] == TEST_MODEL

    async def test_health_check_unhealthy(
        self, connector_with_mock_client: tuple[ClaudeConnector, AsyncMock]
    ) -> None:
        connector, mock_client = connector_with_mock_client
        mock_client.messages.create.side_effect = Exception("Connection failed")

        result = await connector.health_check()

        assert result["status"] == "unhealthy"
        assert "Connection failed" in result["error"]
