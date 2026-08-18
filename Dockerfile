FROM python:3.12-slim

# The project lockfile, rather than a floating pip resolution, is the source
# of truth for application dependencies.
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy the package sources needed while uv installs the locked project.
COPY pyproject.toml uv.lock README.md ./
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# The lock file contains the complete dependency graph.  The application uses
# CPU inference, so omit the CUDA distributions selected by that graph and
# install the same locked torch version from the CPU wheel index.
RUN uv sync --locked --no-dev --no-install-project \
    --no-install-package torch \
    --no-install-package cuda-bindings \
    --no-install-package cuda-pathfinder \
    --no-install-package cuda-toolkit \
    --no-install-package nvidia-cublas \
    --no-install-package nvidia-cuda-cupti \
    --no-install-package nvidia-cuda-nvrtc \
    --no-install-package nvidia-cuda-runtime \
    --no-install-package nvidia-cudnn-cu13 \
    --no-install-package nvidia-cufft \
    --no-install-package nvidia-cufile \
    --no-install-package nvidia-curand \
    --no-install-package nvidia-cusolver \
    --no-install-package nvidia-cusparse \
    --no-install-package nvidia-cusparselt-cu13 \
    --no-install-package nvidia-nccl-cu13 \
    --no-install-package nvidia-nvjitlink \
    --no-install-package nvidia-nvshmem-cu13 \
    --no-install-package nvidia-nvtx \
    --no-install-package triton \
    && uv pip install --python /app/.venv/bin/python \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.13.0+cpu"

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/models \
    && chown appuser:appuser /app/models

# The build context is filtered by .dockerignore, so local credentials,
# generated models, caches, and dataset_builder output cannot enter layers.
COPY --chown=appuser:appuser . .

USER appuser

CMD ["streamlit", "run", "app.py"]
