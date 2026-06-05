#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from PIL import Image, ImageOps
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache frozen VAE latents for extracted image/caption samples."
    )
    parser.add_argument(
        "--caption-embeds-dir",
        default="cache/overfit-emnist-caption-embeds/text-embeds-qwen3-0_6b-bf16",
        help="Directory containing the caption embedding manifest.jsonl.",
    )
    parser.add_argument(
        "--vae",
        default="models/z-image-vae",
        help="Diffusers AutoencoderKL directory with config.json and diffusion_pytorch_model.safetensors.",
    )
    parser.add_argument(
        "--output-dir",
        default="cache/overfit-emnist-vae-latents/z-image-vae-128-bf16",
        help="Directory for latent safetensors and manifest.jsonl.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for VAE encoding. Defaults to cuda and fails fast if CUDA is unavailable.",
    )
    parser.add_argument(
        "--bucket-max-pixels",
        type=int,
        default=16384,
        help="Maximum bucket area in pixels. Default is 1024*768.",
    )
    parser.add_argument("--bucket-min-side", type=int, default=128)
    parser.add_argument("--bucket-max-side", type=int, default=128)
    parser.add_argument("--bucket-step", type=int, default=16)
    parser.add_argument(
        "--crop-mode",
        choices=["center"],
        default="center",
        help="Resize preserving aspect ratio, then crop to selected bucket.",
    )
    parser.add_argument(
        "--sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sample from posterior. Default uses posterior mode for deterministic caching.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--project-root", default=".")
    return parser.parse_args()


def project_relative(path: Path, project_root: Path) -> str:
    path = path.resolve()
    project_root = project_root.resolve()
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def generate_buckets(min_side: int, max_side: int, step: int, max_pixels: int) -> list[tuple[int, int]]:
    buckets = []
    for height in range(min_side, max_side + 1, step):
        for width in range(min_side, max_side + 1, step):
            if height * width <= max_pixels:
                buckets.append((height, width))
    if not buckets:
        raise ValueError("No buckets generated. Check bucket side and pixel limits.")
    return sorted(set(buckets), key=lambda size: (size[0] * size[1], size[0], size[1]))


def choose_bucket(width: int, height: int, buckets: list[tuple[int, int]], target_pixels: int) -> tuple[int, int]:
    image_ratio = width / height

    def score(bucket: tuple[int, int]) -> tuple[float, int]:
        bucket_h, bucket_w = bucket
        bucket_ratio = bucket_w / bucket_h
        ratio_error = abs(math.log(bucket_ratio / image_ratio))
        area_error = abs(math.log((bucket_h * bucket_w) / target_pixels))
        return ratio_error + 0.05 * area_error, -(bucket_h * bucket_w)

    return min(buckets, key=score)


def resize_crop_to_bucket(image: Image.Image, bucket_h: int, bucket_w: int) -> Image.Image:
    width, height = image.size
    scale = max(bucket_w / width, bucket_h / height)
    new_size = (math.ceil(width * scale), math.ceil(height * scale))
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    left = max((image.width - bucket_w) // 2, 0)
    top = max((image.height - bucket_h) // 2, 0)
    return image.crop((left, top, left + bucket_w, top + bucket_h))


def preprocess_image(path: Path, bucket_h: int, bucket_w: int) -> tuple[torch.Tensor, tuple[int, int]]:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    original_size = (image.height, image.width)
    image = resize_crop_to_bucket(image, bucket_h, bucket_w)

    data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    data = data.view(bucket_h, bucket_w, 3).permute(2, 0, 1).float()
    return data.div_(127.5).sub_(1.0), original_size


def iter_batches(items: list[dict], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    caption_embeds_dir = Path(args.caption_embeds_dir)
    output_dir = Path(args.output_dir)
    latent_dir = output_dir / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    weight_dtype = dtype_map[args.dtype]
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this Python process. Run this script from an environment "
            "where torch.cuda.is_available() is true, or pass --device cpu explicitly."
        )
    device = torch.device(args.device)
    print(f"Loading VAE on {device} with dtype={weight_dtype}")

    vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=weight_dtype).to(device)
    vae.eval()
    vae.requires_grad_(False)

    rows = load_manifest(caption_embeds_dir / "manifest.jsonl")
    if args.limit is not None:
        rows = rows[: args.limit]

    buckets = generate_buckets(
        min_side=args.bucket_min_side,
        max_side=args.bucket_max_side,
        step=args.bucket_step,
        max_pixels=args.bucket_max_pixels,
    )
    for row in rows:
        image_path = Path(row["project_relative_image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size
        bucket_h, bucket_w = choose_bucket(width, height, buckets, args.bucket_max_pixels)
        row["_bucket_h"] = bucket_h
        row["_bucket_w"] = bucket_w
        row["_bucket_id"] = f"{bucket_h}x{bucket_w}"

    grouped_rows = defaultdict(list)
    for row in rows:
        grouped_rows[row["_bucket_id"]].append(row)
    ordered_groups = sorted(grouped_rows.items(), key=lambda item: (item[0], len(item[1])))

    manifest_path = output_dir / "manifest.jsonl"
    completed = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for bucket_id, bucket_rows in ordered_groups:
            bucket_h, bucket_w = map(int, bucket_id.split("x"))
            for _, batch_rows in iter_batches(bucket_rows, args.batch_size):
                images = []
                original_sizes = []
                for row in batch_rows:
                    image_path = Path(row["project_relative_image_path"])
                    image_tensor, original_size = preprocess_image(image_path, bucket_h, bucket_w)
                    images.append(image_tensor)
                    original_sizes.append(original_size)

                pixel_values = torch.stack(images, dim=0).to(device=device, dtype=weight_dtype)
                with torch.inference_mode():
                    posterior = vae.encode(pixel_values).latent_dist
                    raw_latents = posterior.sample() if args.sample else posterior.mode()
                    latents = (raw_latents - vae.config.shift_factor) * vae.config.scaling_factor

                for offset, row in enumerate(batch_rows):
                    sample_id = row["id"]
                    cache_file = latent_dir / f"{sample_id}.safetensors"
                    sample_latent = latents[offset].detach().cpu().to(weight_dtype).contiguous()
                    save_file(
                        {
                            "latents": sample_latent,
                            "original_size": torch.tensor(original_sizes[offset], dtype=torch.int64),
                            "bucket_size": torch.tensor([bucket_h, bucket_w], dtype=torch.int64),
                            "latent_size": torch.tensor(sample_latent.shape[-2:], dtype=torch.int64),
                        },
                        str(cache_file),
                    )
                    output_row = {
                        **{k: v for k, v in row.items() if not k.startswith("_")},
                        "latent_file": str(cache_file.relative_to(output_dir)),
                        "project_relative_latent_path": project_relative(cache_file, project_root),
                        "vae": str(args.vae),
                        "vae_project_relative_path": project_relative(Path(args.vae), project_root),
                        "bucket_id": bucket_id,
                        "bucket_size": [bucket_h, bucket_w],
                        "resize_mode": "aspect-ratio-bucket-center-crop",
                        "latent_shape": list(sample_latent.shape),
                        "latent_dtype": args.dtype,
                    }
                    manifest.write(json.dumps(output_row, ensure_ascii=False) + "\n")

                completed += len(batch_rows)
                print(f"[{completed}/{len(rows)}] cached bucket {bucket_id} -> {latent_dir}")

    metadata = {
        "caption_embeds_dir": str(caption_embeds_dir),
        "project_relative_caption_embeds_dir": project_relative(caption_embeds_dir, project_root),
        "vae": str(args.vae),
        "project_relative_vae": project_relative(Path(args.vae), project_root),
        "output_dir": str(output_dir),
        "project_relative_output_dir": project_relative(output_dir, project_root),
        "bucket_mode": "aspect-ratio",
        "bucket_max_pixels": args.bucket_max_pixels,
        "bucket_min_side": args.bucket_min_side,
        "bucket_max_side": args.bucket_max_side,
        "bucket_step": args.bucket_step,
        "bucket_count": len(grouped_rows),
        "bucket_distribution": {bucket_id: len(bucket_rows) for bucket_id, bucket_rows in ordered_groups},
        "resize_mode": "aspect-ratio-bucket-center-crop",
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "posterior": "sample" if args.sample else "mode",
        "latent_scaling": "(raw_latent - shift_factor) * scaling_factor",
        "scaling_factor": float(vae.config.scaling_factor),
        "shift_factor": float(vae.config.shift_factor),
        "num_samples": len(rows),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Done. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
