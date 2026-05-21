FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# System deps - keep minimal
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

# Expose Streamlit port
EXPOSE 8501

# Default to the Streamlit UI; override with `command:` in docker-compose for CLI/eval.
CMD ["streamlit", "run", "src/ui/streamlit_app.py", \
     "--server.address", "0.0.0.0", "--server.port", "8501"]
