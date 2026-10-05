# תמונת בסיס עם CUDA + cuDNN להרצת faster-whisper ו-pyannote על GPU.
# CUDA 12.8 תומך ב-Blackwell/Ada/Hopper/Ampere. Ubuntu 22.04 מביא Python 3.10.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# index של torch ל-CUDA — ניתן לדרוס ב-build: --build-arg CUDA_INDEX=.../cu124 ל-GPU ישן
ARG CUDA_INDEX=https://download.pytorch.org/whl/cu128

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    HF_HOME=/app/data/hf-cache

# Python + ffmpeg + ספריות מערכת (libsndfile ל-soundfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv ffmpeg libsndfile1 git \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# קודם התלויות (pyannote/torchcodec מושכים torch CPU), ואז כופים torch ל-CUDA *אחרון*
# כדי שגרסת ה-GPU תגבר — אחרת ההסקה תרוץ על CPU.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt && \
    pip uninstall -y torch torchaudio && \
    pip install torch torchaudio --index-url ${CUDA_INDEX}

COPY app ./app
COPY web ./web
COPY scripts ./scripts

EXPOSE 8000

# ברירת המחדל מריצה את ה-API. ה-worker מורץ כשירות נפרד ב-compose.
# הערה: הסרת PII (HebSafeHarbor) אינה כלולה בתמונה — דורשת venv מבודד של Python 3.10. ראו DEPLOYMENT.md.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
