#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from train import create_model, load_toml, model_time_from_sigma, resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one checkpoint's flow prediction statistics.")
    parser.add_argument("--config", default="configs/nano-s3dit-overfit-176m.toml")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/nano-s3dit-overfit-176m/checkpoint-002000.pt",
    )
    parser.add_argument("--sample-id", default="1")
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0)


def main() -> int:
    args = parse_args()
    suite_root = Path.cwd()
    config = load_toml(resolve(args.config, suite_root))
    checkpoint = torch.load(resolve(args.checkpoint, suite_root), map_location="cpu")
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
        config = checkpoint["config"]
    manifest = load_manifest(resolve(config["paths"]["manifest"], suite_root))
    row = next((item for item in manifest if str(item["id"]) == str(args.sample_id)), None)
    if row is None:
        raise ValueError(f"sample id {args.sample_id!r} not found")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Pass --device cpu for CPU diagnostics.")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if config["train"].get("dtype", "bf16") == "bf16" else torch.float16
    if device.type == "cpu":
        dtype = torch.float32

    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    model = create_model(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device=device, dtype=dtype)
    model.eval()

    latent = load_file(str(resolve(row["project_relative_latent_path"], suite_root)))["latents"]
    prompt_embeds = load_file(str(resolve(row["project_relative_embedding_path"], suite_root)))["prompt_embeds"]
    x0 = latent.unsqueeze(0).to(device=device, dtype=dtype)
    cap = [prompt_embeds.to(device=device, dtype=dtype)]

    noise = torch.randn_like(x0)
    sigma = torch.full((1,), float(args.sigma), device=device, dtype=torch.float32)
    sigma_view = sigma.to(dtype).view(-1, 1, 1, 1)
    xt = (1.0 - sigma_view) * x0 + sigma_view * noise
    target = noise - x0
    model_time = model_time_from_sigma(sigma, config["train"])

    patch_size = int(config["model"]["patch_size"])
    f_patch_size = int(config["model"]["f_patch_size"])
    with torch.inference_mode():
        raw = model(
            list(xt.unsqueeze(2).unbind(dim=0)),
            model_time,
            cap,
            patch_size=patch_size,
            f_patch_size=f_patch_size,
            return_dict=False,
        )[0]
    pred = -torch.stack([item.squeeze(1) for item in raw], dim=0).to(dtype)
    zero_loss = F.mse_loss(torch.zeros_like(target).float(), target.float())
    pred_loss = F.mse_loss(pred.float(), target.float())

    print(f"checkpoint={resolve(args.checkpoint, suite_root)}")
    print(f"checkpoint_step={checkpoint.get('step')} checkpoint_loss={checkpoint.get('loss')} best_loss={checkpoint.get('best_loss')}")
    print(
        f"sample_id={row['id']} caption={row.get('caption')!r} "
        f"sigma={float(args.sigma):.4f} model_time={model_time.item():.4f} "
        f"model_time_convention={config['train'].get('model_time', 'one_minus_sigma')}"
    )
    print(f"x0 mean={x0.float().mean().item():.6f} std={x0.float().std().item():.6f}")
    print(f"target mean={target.float().mean().item():.6f} std={target.float().std().item():.6f}")
    print(f"pred mean={pred.float().mean().item():.6f} std={pred.float().std().item():.6f}")
    print(f"mse_pred={pred_loss.item():.6f} mse_zero={zero_loss.item():.6f}")
    print(f"cosine_pred_target={cosine_similarity(pred, target).item():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
