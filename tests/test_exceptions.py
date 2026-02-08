"""Tests for custom exceptions."""

import pytest

from job_posting_extractor.exceptions import (
    BusinessError,
    ConfigurationError,
    ExtractionError,
    InputValidationError,
)


class TestBusinessError:
    """Tests for BusinessError base exception."""

    def test_default_values(self) -> None:
        error = BusinessError("Something went wrong")
        assert error.message == "Something went wrong"
        assert error.error_code == "BUSINESS_ERROR"
        assert error.status_code == 400
        assert str(error) == "Something went wrong"

    def test_custom_values(self) -> None:
        error = BusinessError(
            message="Custom error",
            error_code="CUSTOM_CODE",
            status_code=418,
        )
        assert error.message == "Custom error"
        assert error.error_code == "CUSTOM_CODE"
        assert error.status_code == 418

    def test_is_exception_subclass(self) -> None:
        error = BusinessError("test")
        assert isinstance(error, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(BusinessError) as exc_info:
            raise BusinessError("Test message")
        assert exc_info.value.message == "Test message"


class TestDerivedExceptions:
    """Tests for BusinessError derived exceptions."""

    @pytest.mark.parametrize(
        ("exception_class", "error_code", "status_code"),
        [
            (ExtractionError, "EXTRACTION_ERROR", 422),
            (InputValidationError, "INPUT_VALIDATION_ERROR", 400),
            (ConfigurationError, "CONFIGURATION_ERROR", 500),
        ],
        ids=["ExtractionError", "InputValidationError", "ConfigurationError"],
    )
    def test_default_values(
        self,
        exception_class: type[BusinessError],
        error_code: str,
        status_code: int,
    ) -> None:
        error = exception_class("Test message")
        assert error.message == "Test message"
        assert error.error_code == error_code
        assert error.status_code == status_code

    @pytest.mark.parametrize(
        "exception_class",
        [ExtractionError, InputValidationError, ConfigurationError],
    )
    def test_is_business_error_subclass(
        self, exception_class: type[BusinessError]
    ) -> None:
        error = exception_class("test")
        assert isinstance(error, BusinessError)

    def test_extraction_error_can_be_caught_as_business_error(self) -> None:
        with pytest.raises(BusinessError) as exc_info:
            raise ExtractionError("Extraction failed")
        assert exc_info.value.error_code == "EXTRACTION_ERROR"
