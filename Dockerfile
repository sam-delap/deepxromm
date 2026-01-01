FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    python3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/bin/" sh

ENV UV_SYSTEM_PYTHON=1
ENV UV_LINK_MODE=copy

# hadolint ignore=DL3045
COPY pyproject.toml .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r pyproject.toml --group dev

ENTRYPOINT ["/bin/bash"]
