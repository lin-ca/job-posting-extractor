"""Custom exceptions for the application."""


class BusinessError(Exception):
    """Base exception for application-specific business logic errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "BUSINESS_ERROR",
        status_code: int = 400,
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(self.message)


class ExtractionError(BusinessError):
    """Raised when data extraction fails."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            error_code="EXTRACTION_ERROR",
            status_code=422,
        )


class InputValidationError(BusinessError):
    """Raised when input validation fails."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            error_code="INPUT_VALIDATION_ERROR",
            status_code=400,
        )


class ConfigurationError(BusinessError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            status_code=500,
        )
