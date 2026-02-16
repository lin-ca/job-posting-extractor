FROM python:3.12-slim

RUN useradd --create-home appuser

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files and README (required by hatchling build backend)
COPY pyproject.toml uv.lock README.md ./

# Copy source code
COPY src/ src/

# Install dependencies and project
RUN uv sync --frozen && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

EXPOSE 8000
CMD [".venv/bin/job-posting-extractor"]
