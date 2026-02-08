# Job Posting Extractor

A modern FastAPI application that extracts structured job posting data from unstructured text using Claude AI.

Side project to try out different design patterns and a few newer Python and FastAPI features (e.g. lifespan context manager, application factory, ... ).


## Features

- FastAPI web framework
- Modern Python tooling: [uv](https://docs.astral.sh/uv/) for fast dependency management, [Ruff](https://docs.astral.sh/ruff/) for linting/formatting, [mypy](https://mypy.readthedocs.io/) for strict type checking
- Claude AI integration using the Anthropic Python SDK with tool use
- Environment-based configuration with pydantic-settings
- Interactive API documentation (Swagger UI and ReDoc)
- Mock connector for testing without API calls
- Unit and integration test suite covering models, API endpoints, connectors, and services


## Prerequisites

- Python 3.12+
- Anthropic API key ([get one here](https://console.anthropic.com/))

## Setup

1. **Clone and navigate to the project:**
   ```bash
   cd job-posting-extractor
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your_actual_api_key_here
   ```

## Running the Application

Start the FastAPI server:

```bash
make run
```

Or with auto-reload for development:

```bash
make dev
```

The application will be available at `http://localhost:8000`

## API Endpoints

### GET `/health`
Health check endpoint.

```bash
curl http://localhost:8000/health
```

### POST `/api/v1/extract/job`
Extract structured data from job posting text.

```bash
curl -X POST http://localhost:8000/api/v1/extract/job \
  -H "Content-Type: application/json" \
  -d '{"text": "Senior Python Developer at TechCorp - Berlin (Hybrid)..."}'
```

## Interactive API Documentation

FastAPI provides automatic interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Configuration

Environment variables (with defaults):

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key | (required) |
| `CLAUDE_MODEL` | Claude model to use | `claude-sonnet-4-5-20250929` |
| `MAX_TOKENS` | Maximum tokens for responses | `1024` |
| `API_TIMEOUT` | API request timeout in seconds | `60.0` |
| `MOCK_LLM` | Whether or not to mock LLM call | `false` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `RELOAD` | Enable auto-reload | `false` |
| `LOG_LEVEL` | Logging level | `info` |

## Development

Install development dependencies:

```bash
make install
```

Available Make targets (run `make help` to see all):

| Command | Description |
|---------|-------------|
| `make run` | Start the API server |
| `make dev` | Start the API server with auto-reload |
| `make lint` | Run linting checks |
| `make format` | Format code |
| `make typecheck` | Run type checking |
| `make test` | Run tests |
| `make ci` | Run all checks (lint, typecheck, test) |
| `make clean` | Remove build artifacts and caches |

## Architecture

```
src/job_posting_extractor/
├── __init__.py              # Version management
├── config.py                # Pydantic settings (singleton via @cache)
├── models.py                # Pydantic v2 request/response models
├── exceptions.py            # Custom exception hierarchy
├── connectors/              # External API adapters
│   ├── base.py              # Protocol definitions (interfaces)
│   ├── claude.py            # Production Claude API connector
│   └── mock_claude.py       # Mock connector for testing
├── services/                # Business logic layer
│   └── extraction.py        # Orchestrates extraction + confidence
└── api/                     # FastAPI application layer
    ├── service.py           # App factory + lifespan management
    ├── dependencies.py      # Dependency injection setup
    └── routers/
        └── extraction.py    # Job extraction endpoints
```