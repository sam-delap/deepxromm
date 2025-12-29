FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    git \
    libgl1 \
    libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /venv
COPY pyproject.toml uv.lock /venv/
WORKDIR /venv

RUN uv sync

ENTRYPOINT ["/bin/bash"]
