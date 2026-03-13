# bitmind-subnet Validator stack image.
# One image; run as validator, generator, or data via SERVICE=validator|generator|data.
# Compose starts three containers with shared volumes.

# ---------------------------------------------------------------------------
# Build stage: install all Python dependencies
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1

# System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    cmake \
    git \
    wget \
    curl \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock VERSION ./

# Create venv and install dependencies (frozen = use lockfile exactly)
RUN uv sync --frozen

# Copy full project source
COPY . .

# Install the gas package (editable) and additional git dependencies.
# One of these (Janus/diffusers/CLIP) uses setuptools at build time without declaring it,
# so we install setuptools and use --no-build-isolation for these installs only. In Docker
# this is safe: the build env is fixed (only uv sync + setuptools), so the build is
# reproducible. uv's extra-build-dependencies does not work reliably with setuptools.
RUN uv pip install setuptools && \
    uv pip install -e . && \
    uv pip install --no-build-isolation \
        git+https://github.com/deepseek-ai/Janus.git \
        git+https://github.com/huggingface/diffusers \
        git+https://github.com/openai/CLIP.git

# ---------------------------------------------------------------------------
# Runtime stage: slim image with only what's needed to run
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Runtime system dependencies (gcc + python3.10-dev for Triton/bitsandbytes JIT C compile at import)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    build-essential \
    ffmpeg \
    xvfb \
    wget \
    gnupg2 \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libxshmfence1 \
    && rm -rf /var/lib/apt/lists/*

# Install Google Chrome (needed by data service Selenium scraper)
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - && \
    echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" \
        > /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends google-chrome-stable && \
    rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

WORKDIR /app

# Copy entire project (including .venv and docker/) from builder
COPY --from=builder /app /app
RUN chmod +x /app/docker/entrypoint.sh

# Ensure cache directories exist
RUN mkdir -p /root/.cache/sn34/tmp /root/.cache/huggingface /root/.bittensor/wallets

# Default environment
ENV PATH="/app/.venv/bin:$PATH"
ENV SN34_CACHE_DIR=/root/.cache/sn34
ENV HF_HOME=/root/.cache/huggingface
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV TRANSFORMERS_VERBOSITY=error
ENV DIFFUSERS_VERBOSITY=error
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HUB_VERBOSITY=error
ENV ACCELERATE_LOG_LEVEL=error
ENV TMPDIR=/root/.cache/sn34/tmp
ENV TEMP=/root/.cache/sn34/tmp
ENV TMP=/root/.cache/sn34/tmp
ENV DISPLAY=:99

ENTRYPOINT ["/app/docker/entrypoint.sh"]
