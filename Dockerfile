# Use Python 3.11 image with ubuntu base
FROM python:3.11

WORKDIR /app

# Create non-root user for security
RUN groupadd -r app && useradd -r -g app app

# Set pip configuration for better reliability
ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install base requirements first
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --retries 5

# Copy application code
COPY . .

# Create required directories with proper permissions
RUN mkdir -p /app/logs /app/data \
    && chown -R app:app /app

# Switch to non-root user
USER app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Command to run the FastAPI server
CMD ["uvicorn", "simple_api:app", "--host", "0.0.0.0", "--port", "8000"]

# Run the application
CMD ["uvicorn", "simple_api:app", "--host", "0.0.0.0", "--port", "8000"]
