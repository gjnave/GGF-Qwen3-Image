"""One-shot Qwen-Image text-to-image worker using the app-local Comfy core."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PIL import Image

from model_store import ENCODER_FILE, MODEL_DIR, VAE_FILE, gguf_name, verify

APP = Path(__file__).resolve().parent
CORE = APP / "vendor" / "comfy_core"
VENDOR = APP / "vendor"


def emit(**fields):
    print(json.dumps(fields), flush=True)


def main(request_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    quant = str(request["quant"])
    ready, message = verify(quant)
    if not ready:
        raise RuntimeError(message)
    if not (CORE / "nodes.py").is_file() or not (VENDOR / "comfy_gguf" / "nodes.py").is_file():
        raise RuntimeError("App-local Comfy core or GGUF loader is missing. Reinstall the app source.")
    emit(status=f"Loading Qwen-Image {quant} with the app-local Comfy core…")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("An NVIDIA GPU with a compatible CUDA driver is required.")
    width, height = int(request["width"]), int(request["height"])
    if not (256 <= width <= 4096 and 256 <= height <= 4096 and width % 32 == 0 and height % 32 == 0):
        raise ValueError("Width and height must be 256–4096 pixels in multiples of 32.")
    if not str(request.get("prompt", "")).strip():
        raise ValueError("Describe the image you want to create.")
    os.chdir(CORE)
    sys.path.insert(0, str(CORE))
    sys.path.insert(0, str(VENDOR))
    sys.argv = [sys.argv[0], "--models-directory", str(MODEL_DIR)]
    import folder_paths
    import nodes
    folder_paths.add_model_folder_path("diffusion_models", str(MODEL_DIR), is_default=True)
    folder_paths.add_model_folder_path("text_encoders", str(MODEL_DIR / "split_files" / "text_encoders"), is_default=True)
    folder_paths.add_model_folder_path("vae", str(MODEL_DIR / "split_files" / "vae"), is_default=True)
    from comfy_gguf.nodes import UnetLoaderGGUF
    from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow

    model = UnetLoaderGGUF().load_unet(gguf_name(quant))[0]
    clip = nodes.CLIPLoader().load_clip(Path(ENCODER_FILE).name, "qwen_image", "default")[0]
    vae = nodes.VAELoader().load_vae(Path(VAE_FILE).name)[0]
    model = ModelSamplingAuraFlow().patch_aura(model, 3.1)[0]
    emit(status="Model loaded. Encoding your prompt…")
    with torch.no_grad():
        positive = nodes.CLIPTextEncode().encode(clip, request["prompt"])[0]
        negative = nodes.CLIPTextEncode().encode(clip, request.get("negative_prompt") or "")[0]
        import comfy.model_management
        latent = {"samples": torch.zeros(
            [1, 16, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
            dtype=comfy.model_management.intermediate_dtype()),
            "downscale_ratio_spacial": 8}
    seed = int(request["seed"])
    if seed < 0:
        seed = int.from_bytes(os.urandom(4), "little")
    import latent_preview
    original_callback_factory = latent_preview.prepare_callback

    def progress_callback_factory(model_for_preview, total_steps, x0_output_dict=None):
        base_callback = original_callback_factory(model_for_preview, total_steps, x0_output_dict)

        def report_step(step, x0, x, count):
            base_callback(step, x0, x, count)
            emit(step=step + 1, total=count,
                 status=f"Generating · step {step + 1} of {count} · {quant}")

        return report_step

    latent_preview.prepare_callback = progress_callback_factory
    emit(status=f"Generating {width} × {height} image · {request['steps']} steps with {quant}…")
    with torch.no_grad():
        sampled = nodes.KSampler().sample(
            model=model, seed=seed, steps=int(request["steps"]), cfg=float(request["cfg"]),
            sampler_name="euler", scheduler="simple", positive=positive, negative=negative,
            latent_image=latent, denoise=1.0)[0]
        pixels = nodes.VAEDecode().decode(vae, sampled)[0][0]
    import numpy as np
    result = Image.fromarray((pixels.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8))
    output = Path(request["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    emit(complete=str(output), seed=seed, quant=quant)


if __name__ == "__main__":
    try:
        main(Path(sys.argv[1]))
    except Exception as exc:
        emit(error=f"{type(exc).__name__}: {exc}")
        raise SystemExit(1)
