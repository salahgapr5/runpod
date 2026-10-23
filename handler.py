import os

# Cache model weights on the network volume (mounted at /runpod-volume on serverless)
if os.path.isdir("/runpod-volume"):
    os.environ.setdefault("HF_HOME", "/runpod-volume/hf")

import base64
import inspect
import io
import traceback

import runpod
import torch
from PIL import Image

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen-Image-2.1-Turbo")
OFFLOAD = os.environ.get("OFFLOAD", "0") == "1"  # set OFFLOAD=1 if you get out-of-memory

pipe = None
accepted = set()


def load_pipe():
    global pipe, accepted
    from diffusers import QwenImage21Pipeline

    print(f"Loading {MODEL_ID} (HF_HOME={os.environ.get('HF_HOME')})", flush=True)
    pipe = QwenImage21Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    if OFFLOAD:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    accepted = set(inspect.signature(pipe.__call__).parameters.keys())
    print("Pipeline ready. Accepted call args:", sorted(accepted), flush=True)


def handler(job):
    try:
        if pipe is None:
            load_pipe()

        inp = job.get("input", {})
        prompt = inp.get("prompt")
        if not prompt:
            return {"error": "Missing 'prompt' in input"}

        kwargs = {"prompt": prompt}

        # Optional args: only passed if the pipeline accepts them
        if "width" in accepted and inp.get("width"):
            kwargs["width"] = int(inp["width"])
        if "height" in accepted and inp.get("height"):
            kwargs["height"] = int(inp["height"])
        # Turbo has its 8-step schedule saved with the weights, so steps are only
        # passed if you explicitly ask for them.
        if "num_inference_steps" in accepted and inp.get("num_inference_steps"):
            kwargs["num_inference_steps"] = int(inp["num_inference_steps"])
        if "negative_prompt" in accepted and inp.get("negative_prompt"):
            kwargs["negative_prompt"] = inp["negative_prompt"]
        if "generator" in accepted and inp.get("seed") is not None:
            kwargs["generator"] = torch.Generator("cuda").manual_seed(int(inp["seed"]))

        image = pipe(**kwargs).images[0]

        # Optional resize, e.g. "resize_to": [1920, 1080]
        resize_to = inp.get("resize_to")
        if resize_to:
            image = image.resize((int(resize_to[0]), int(resize_to[1])), Image.LANCZOS)

        fmt = inp.get("output_format", "png").lower()
        fmt = "JPEG" if fmt in ("jpg", "jpeg") else "PNG"
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format=fmt, quality=95)

        return {
            "image_base64": base64.b64encode(buf.getvalue()).decode("utf-8"),
            "format": fmt.lower(),
            "width": image.width,
            "height": image.height,
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}


runpod.serverless.start({"handler": handler})
