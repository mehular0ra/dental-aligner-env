FROM python:3.11-slim

# Install uv — the fast Python package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency manifests first for optimal layer caching.
# Dependencies are resolved from pyproject.toml using the pinned uv.lock.
COPY pyproject.toml uv.lock ./

# Install all runtime dependencies without the project itself.
# --frozen: refuse to update uv.lock (reproducible builds).
# --no-dev: skip dev/test extras.
RUN uv sync --frozen --no-dev --no-install-project

# Copy the rest of the source tree.
COPY . .

# Install the project package into the same virtual environment.
RUN uv sync --frozen --no-dev

EXPOSE 7860

# Use uv run so the project virtualenv is activated automatically.
CMD ["uv", "run", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
