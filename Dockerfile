# Use official Python runtime as base image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for plots and logs
RUN mkdir -p App/static/plots logs

# Set environment variables
ENV FLASK_DEBUG=false
ENV PYTHONPATH=/app

# Expose ports for both applications
EXPOSE 5000 5001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5001/health || exit 1

# Default command - run both applications
CMD ["./run.sh", "both"]