# ===============================
# Base image (production)
# ===============================
FROM python:3.11-slim

# ===============================
# System dependencies (LightGBM)
# ===============================
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ===============================
# Working directory
# ===============================
WORKDIR /app

# ===============================
# Install Python dependencies
# ===============================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ===============================
# Copy source code & models
# ===============================
COPY src ./src
COPY models ./models

# ===============================
# Environment variables
# ===============================
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# ===============================
# Expose port
# ===============================
# EXPOSE 8000

# ===============================
# Run FastAPI with Uvicorn
# ===============================
CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port $PORT"]

