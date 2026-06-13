# Multi-stage build — production image < 500MB
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user \
    torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --user \
    torch-geometric==2.5.3 && \
    pip install --no-cache-dir --user \
    -r requirements.txt


FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Non-root user for security
RUN useradd --create-home pragma && chown -R pragma:pragma /app
USER pragma

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
