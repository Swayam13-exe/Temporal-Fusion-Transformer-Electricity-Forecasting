FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first -- the base image's default pip (23.0.1) has a known bug
# parsing PyTorch's custom package index metadata (underscore vs hyphen name
# normalization), which breaks the CPU-only torch install below otherwise.
RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
RUN grep -v '^torch==' requirements.txt > requirements-container.txt && \
    pip install --no-cache-dir -r requirements-container.txt

COPY src/ src/
COPY config.yaml .
COPY models/ models/
COPY demo/ demo/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]