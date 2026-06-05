#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

from cache_vae_latents import resize_crop_to_bucket


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode one cached VAE latent and compare it with its source image.")
    parser.add_argument(
        "--manifest",
        default="cache/overfit-emnist-vae-latents/z-image-vae-128-bf16/manifest.jsonl",
    )
    parser.add_argument("--sample-id", default="1")
    parser.add_argument("--vae", default="models/z-image-vae")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        default="outputs/debug-vae-reconstruct-sample-1.png",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().float().cpu().clamp(0.0, 1.0)
    image = (image * 255.0).round().to(torch.uint8)
    image = image.permute(1, 2, 0).numpy()
    return Image.fromarray(image)


def main() -> int:
    args = parse_args()
    suite_root = Path.cwd()
    manifest_path = Path(args.manifest)
    rows = load_manifest(manifest_path)
    row = next((item for item in rows if str(item["id"]) == str(args.sample_id)), None)
    if row is None:
        raise ValueError(f"sample id {args.sample_id!r} not found in {manifest_path}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Pass --device cpu to run this debug script on CPU.")
    device = torch.device(args.device)

    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=torch.float32).to(device)
    vae.eval()
    vae.requires_grad_(False)

    latent_path = suite_root / row["project_relative_latent_path"]
    image_path = suite_root / row["project_relative_image_path"]
    latent = load_file(str(latent_path))["latents"].unsqueeze(0).to(device=device, dtype=torch.float32)
    decode_latent = (latent / vae.config.scaling_factor) + vae.config.shift_factor

    with torch.inference_mode():
        decoded = vae.decode(decode_latent, return_dict=False)[0][0]
    decoded = (decoded / 2.0 + 0.5).clamp(0.0, 1.0)

    bucket_h, bucket_w = row["bucket_size"]
    source = Image.open(image_path).convert("RGB")
    source = resize_crop_to_bucket(source, int(bucket_h), int(bucket_w))

    decoded_image = tensor_to_pil(decoded)
    canvas = Image.new("RGB", (source.width + decoded_image.width, max(source.height, decoded_image.height)), "black")
    canvas.paste(source, (0, 0))
    canvas.paste(decoded_image, (source.width, 0))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

    mse = torch.mean((decoded - torch.from_numpy(__import__("numpy").array(source)).permute(2, 0, 1).float() / 255.0) ** 2)
    print(f"sample_id={row['id']} caption={row.get('caption')!r}")
    print(f"latent_path={latent_path}")
    print(f"image_path={image_path}")
    print(f"latent_shape={tuple(latent.shape)} latent_mean={latent.mean().item():.6f} latent_std={latent.std().item():.6f}")
    print(f"vae_scaling_factor={vae.config.scaling_factor} vae_shift_factor={vae.config.shift_factor}")
    print(f"reconstruction_mse_0_1={mse.item():.6f}")
    print(f"saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
