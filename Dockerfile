# Start from a small Python base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system deps (optional but common)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Copy requirements first to use caching
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# For testing: run a simple Python file
CMD ["python", "src/app.py"]
