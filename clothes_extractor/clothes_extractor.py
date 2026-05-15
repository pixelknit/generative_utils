"""
Extract clothes from an image using FLUX.1-Kontext-dev + the
extract-clothes LoRA from ovi054.

Usage:
    python clothes_extractor.py input.jpg output.png
"""

import argparse
import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

MODEL_PATH = "extract-clothes-kontext-dev-lora.safetensors"

def build_pipeline(lora_repo: str = "ovi054/extract-clothes-kontext-dev-lora"):
    #check devices
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.bfloat16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"Loading FLUX.1-Kontext-dev on {device} ({dtype})...")
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=dtype,
    ).to(device)

    # pipe.enable_model_cpu_offload()

    print(f"Loading LoRA weights from {lora_repo}...")
    pipe.load_lora_weights(lora_repo)

    return pipe


def extract_clothes(
    pipe,
    input_path: str,
    output_path: str,
    prompt: str = "extract only the clothes over a plain background, product photography style",
    guidance_scale: float = 2.5,
    num_inference_steps: int = 28,
    seed: int | None = 42,
):
    input_image = load_image(input_path)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    result = pipe(
        image=input_image,
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]

    result.save(output_path)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path or URL to the input image")
    parser.add_argument("output", help="Where to save the extracted-clothes image")
    parser.add_argument(
        "--prompt",
        default="extract only the clothes over a plain background, product photography style",
    )
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pipe = build_pipeline(lora_repo=MODEL_PATH)
    # pipe = build_pipeline()
    extract_clothes(
        pipe,
        input_path=args.input,
        output_path=args.output,
        prompt=args.prompt,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
