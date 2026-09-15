FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY sample_docs ./sample_docs
RUN uv sync --frozen --no-dev

ENV RAGKB_HOST=0.0.0.0 \
    RAGKB_PORT=8000 \
    RAGKB_CHROMA_DIR=/data/chroma \
    PYTHONUNBUFFERED=1

VOLUME ["/data/chroma"]

CMD ["uv", "run", "python", "-m", "rag_kb.server"]
