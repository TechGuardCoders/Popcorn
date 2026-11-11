# Popcorn — numeric-aware text normalization benchmark container
# Build:  docker build -t popcorn .
# Run:    docker run popcorn   # correctness gate + benchmark, writes benchmarks/

FROM python:3.12-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy
COPY scripts/extract_c.py .
COPY popcorn.py .
# extract the C source and compile the extension against this interpreter
RUN python extract_c.py \
    && INC=$(python -c 'import sysconfig; print(sysconfig.get_paths()["include"])') \
    && gcc -O2 -shared -fPIC -I"$INC" popcorn_c.c \
       -o popcorn_c$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

FROM python:3.12-slim AS runner
WORKDIR /app
RUN pip install --no-cache-dir numpy
COPY --from=builder /build/popcorn.py .
COPY --from=builder /build/popcorn_c* .
ENV PYTHONUNBUFFERED=1
CMD ["python", "popcorn.py"]
