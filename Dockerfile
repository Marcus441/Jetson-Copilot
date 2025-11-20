FROM dustynv/l4t-text-generation:r36.2.0

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/
COPY README.md .

RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir --verbose .

EXPOSE 8000

CMD ["start"]

