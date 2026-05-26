# Dockerfile — Flask Backend
# ============================
# Build:   docker build -t traffic-ai-backend .
# Run:     docker run -p 5000:5000 traffic-ai-backend

FROM python:3.11-slim

# System dependencies for OpenCV, PyTorch CPU, and SQLite
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-caching friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create upload dir
RUN mkdir -p backend/uploads ml_models/weights

# Expose Flask port
EXPOSE 5000

# Environment
ENV FLASK_APP=backend.app
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/health')"

# Run with gunicorn (4 workers)
CMD ["gunicorn", \
     "--workers", "4", \
     "--bind", "0.0.0.0:5000", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "backend.app:create_app()"]
