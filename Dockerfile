# ─────────────────────────────────────────────────────────────
# MAITRI – Production Dockerfile
# ─────────────────────────────────────────────────────────────
FROM python:3.10-slim

# 1. Install system-level dependencies for audio and video
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libasound2-dev \
    gcc \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Set working directory
WORKDIR /app

# 3. Copy only requirements first for faster caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the application
COPY . .

# 5. Create data directory
RUN mkdir -p data

# 6. Expose ports
# 8000: Backend API
# 8501: Streamlit Frontend
EXPOSE 8000
EXPOSE 8501

# 7. Start script (default is backend)
# You should really use docker-compose to run both
# But for single run, we provide a combined cmd
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0"]
