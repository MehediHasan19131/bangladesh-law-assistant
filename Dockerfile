FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf TOKENIZERS_PARALLELISM=false
WORKDIR /app

# CPU-only PyTorch keeps the image a few GB smaller than the default CUDA build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY data ./data
COPY static ./static

# Embed the corpus at build time so the container starts ready to answer
ARG EMBEDDER=BAAI/bge-m3
RUN python -m lawrag.index --embedder "$EMBEDDER" --out index

EXPOSE 8000
CMD ["uvicorn", "lawrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
