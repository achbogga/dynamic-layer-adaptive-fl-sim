FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

# Install sudo and other dev tools, create vscode user
# trunk-ignore(hadolint/DL3008)
RUN apt-get update && \
    apt-get install -y \
        git \
        curl \
        --no-install-recommends && \
    groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER $USERNAME

WORKDIR /workspace

# Set uv cache directory inside workspace to enable hardlinking (see Astral UV docs)
ENV UV_CACHE_DIR=/workspace/.uv-cache

RUN mkdir -p /workspace/.uv-cache

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:80/ || exit 1
