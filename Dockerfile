# Base image with CUDA + cuDNN for running faster-whisper and pyannote on GPU.
# CUDA 12.8 supports Blackwell/Ada/Hopper/Ampere. Ubuntu 22.04 brings Python 3.10.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# index of torch for CUDA — can override in build: --build-arg CUDA_INDEX=.../cu124 for old GPU
ARG CUDA_INDEX=https://download.pytorch.org/whl/cu128

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    HF_HOME=/app/data/hf-cache

# Python + ffmpeg + system libraries (libsndfile for soundfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv ffmpeg libsndfile1 git \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# First dependencies (pyannote/torchcodec pulls torch CPU), then push torch to CUDA *last*
# so the GPU version can upgrade — otherwise the inference will run on CPU.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt && \
    pip uninstall -y torch torchaudio && \
    pip install torch torchaudio --index-url ${CUDA_INDEX}

COPY app ./app
COPY web ./web

EXPOSE 8000

# Default runs the API. The worker runs as a separate service in compose.
# Note: PII removal (HebSafeHarbor) is not included in the image — requires isolated Python 3.10 venv. See DEPLOYMENT.md.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
