FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/hf

RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Turbo needs a Diffusers build with QwenImage21Pipeline (PR #14950 per the model card)
RUN pip install --no-cache-dir \
    "git+https://github.com/huggingface/diffusers.git@refs/pull/14950/head" \
    transformers accelerate safetensors sentencepiece pillow runpod

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
