# ============================================
# FIN-SIGHT AI DOCKERFILE
# Multi-stage build for production-ready container
# ============================================

# STAGE 1: Base Python Image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies (for pgvector/psycopg2)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY data/ ./data/

# Expose port for MLflow UI (optional, for dashboard)
EXPOSE 5000

# Default command: Run interactive RAG mode
CMD ["python", "src/rag_pipeline.py"]