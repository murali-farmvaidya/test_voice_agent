FROM python:3.11-slim

# System dependencies for audio / WebRTC
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install uv
RUN pip install --no-cache-dir uv

# Install dependencies
RUN uv sync --locked --no-dev

# Copy application code
COPY bot.py bot.py
COPY resource_document.txt resource_document.txt

CMD ["python", "bot.py", "--transport", "daily"]
