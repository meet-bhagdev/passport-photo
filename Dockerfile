# Passport Photo Editor - Production Docker Image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PIL/image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py .
COPY static/ static/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with gunicorn (production WSGI server)
# Using 1 worker to keep session data in memory, with more threads for concurrency
CMD ["gunicorn", "--workers=1", "--threads=8", "--bind=0.0.0.0:8000", "--timeout=120", "server:app"]
