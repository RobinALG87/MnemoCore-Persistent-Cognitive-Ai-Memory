# MnemoCore Dockerfile
# ====================
# Multi-stage build for optimized production image

# Build argument for version (can be overridden in CI)
ARG VERSION=2.0.0

# Stage 1: Builder
FROM python:3.11.8-slim-bookworm AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Production
FROM python:3.11.8-slim-bookworm AS production

# Re-declare ARG in production stage to use it
ARG VERSION=2.0.0

# Labels for container metadata
LABEL maintainer="MnemoCore Team"
LABEL description="MnemoCore - Infrastructure for Persistent Cognitive Memory"
LABEL version="${VERSION}"

# Security: Create non-root user
RUN groupadd --gid 1000 mnemocore && \
    useradd --uid 1000 --gid mnemocore --shell /bin/bash --create-home mnemocore

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy application code
COPY --chown=mnemocore:mnemocore src/ ./src/
COPY --chown=mnemocore:mnemocore config.yaml .
COPY --chown=mnemocore:mnemocore scripts/ ./scripts/

# Copy and set up entrypoint script
COPY scripts/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create data directory with proper permissions
RUN mkdir -p /app/data && chown -R mnemocore:mnemocore /app/data

# Switch to non-root user
USER mnemocore

# Environment variables (defaults, can be overridden)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HAIM_API_KEY="" \
    REDIS_URL="redis://redis:6379/0" \
    QDRANT_URL="http://qdrant:6333" \
    LOG_LEVEL="INFO" \
    HOST="0.0.0.0" \
    PORT="8100"

# Expose port
EXPOSE 8100

# Health check using the healthcheck script
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python /app/scripts/ops/healthcheck.py || exit 1

# Entry point: Validate environment then run uvicorn
ENTRYPOINT ["/entrypoint.sh", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8100"]
CMD ["--workers", "1", "--log-level", "info"]
