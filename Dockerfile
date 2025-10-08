FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/home/app/.local/bin:${PATH}"

WORKDIR /app

# Install system deps required by PDF parsing stack (camelot, pdfminer, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ghostscript \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Cache-able layer: install python deps
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . ./

# Create non-root user
RUN useradd --create-home app && chown -R app:app /app
USER app

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["--help"]
