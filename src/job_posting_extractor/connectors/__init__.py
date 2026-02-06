"""Connectors module for external service integrations."""

from job_posting_extractor.connectors.base import Connector, JobExtractor
from job_posting_extractor.connectors.claude import ClaudeConnector

__all__ = [
    "Connector",
    "JobExtractor",
    "ClaudeConnector",
]
