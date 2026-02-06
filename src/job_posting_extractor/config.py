"""Configuration settings for the application using pydantic-settings."""

from functools import cache

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application settings
    app_name: str = "Job Posting Extractor"
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: str = "info"

    # Anthropic API settings
    anthropic_api_key: SecretStr | None = None
    claude_model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 1024
    api_timeout: float = 60.0

    # Mock settings
    mock_llm: bool = False

    @model_validator(mode="after")
    def validate_api_key_when_needed(self) -> "Settings":
        """Validate that the API key is configured when not using mock."""
        if not self.mock_llm and (
            self.anthropic_api_key is None
            or not self.anthropic_api_key.get_secret_value()
        ):
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required when mock_llm is False. "
                "Please set it in your .env file or environment."
            )
        return self

    @property
    def api_key(self) -> SecretStr:
        """Get the API key. Only valid when mock_llm is False."""
        assert self.anthropic_api_key is not None, "API key required when not mocking"
        return self.anthropic_api_key


@cache
def get_settings() -> Settings:
    """Lazy singleton for application settings."""
    return Settings()
