#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset, Sampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-GPU Z-Image-like flow matching pretraining.")
    parser.add_argument("--config", default="configs/nano-s3dit-overfit-176m.toml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def resolve(path: str | Path, root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class ZImageLatentDataset(Dataset):
    def __init__(self, manifest_path: Path, root: Path, limit: int | None = None):
        self.root = root
        self.rows = load_manifest(manifest_path)
        if limit is not None:
            self.rows = self.rows[:limit]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        latent_path = resolve(row["project_relative_latent_path"], self.root)
        embed_path = resolve(row["project_relative_embedding_path"], self.root)
        latent = load_file(str(latent_path))["latents"]
        embeds = load_file(str(embed_path))["prompt_embeds"]
        return {
            "id": row["id"],
            "bucket_id": row["bucket_id"],
            "latents": latent,
            "prompt_embeds": embeds,
        }


class BucketBatchSampler(Sampler[list[int]]):
    def __init__(self, rows: list[dict[str, Any]], batch_size: int, seed: int = 42, drop_last: bool = False):
        self.groups: dict[str, list[int]] = defaultdict(list)
        for idx, row in enumerate(rows):
            self.groups[row["bucket_id"]].append(idx)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches = []
        for indices in self.groups.values():
            indices = indices[:]
            rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        rng.shuffle(batches)
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        total = 0
        for indices in self.groups.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += math.ceil(len(indices) / self.batch_size)
        return total


def collate_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "ids": [sample["id"] for sample in samples],
        "bucket_id": samples[0]["bucket_id"],
        "latents": torch.stack([sample["latents"] for sample in samples], dim=0),
        "prompt_embeds": [sample["prompt_embeds"] for sample in samples],
    }


def create_model(config: dict[str, Any]):
    from nano_s3dit.models import build_model_from_config

    return build_model_from_config(config["model"])


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def cuda_memory_summary(device: torch.device) -> str:
    if device.type != "cuda":
        return "cuda_mem=NA"
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    return f"cuda_alloc={allocated:.2f}G cuda_reserved={reserved:.2f}G cuda_peak={peak:.2f}G"


def learning_rate_for_step(step: int, train_cfg: dict[str, Any]) -> float:
    base_lr = float(train_cfg["learning_rate"])
    min_lr = float(train_cfg.get("min_learning_rate", 0.0))
    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    max_steps = int(train_cfg["max_steps"])
    scheduler = str(train_cfg.get("lr_scheduler", "constant")).lower()

    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps

    if scheduler == "constant":
        return base_lr
    if scheduler != "cosine":
        raise ValueError(f"Unsupported lr_scheduler={scheduler!r}. Expected 'constant' or 'cosine'.")

    decay_steps = max(max_steps - warmup_steps, 1)
    decay_step = min(max(step - warmup_steps, 0), decay_steps)
    progress = decay_step / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def model_time_from_sigma(sigma: torch.Tensor, train_cfg: dict[str, Any]) -> torch.Tensor:
    convention = str(train_cfg.get("model_time", "one_minus_sigma")).lower()
    if convention == "sigma":
        return sigma
    if convention in {"one_minus_sigma", "1-sigma"}:
        return 1.0 - sigma
    raise ValueError(f"Unsupported model_time={convention!r}. Expected 'sigma' or 'one_minus_sigma'.")


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def prune_checkpoints(output_dir: Path, keep_last: int, best_checkpoint: Path | None) -> None:
    if keep_last <= 0 and best_checkpoint is None:
        return

    checkpoints = sorted(
        [path for path in output_dir.glob("checkpoint-*.pt") if checkpoint_step(path) >= 0],
        key=checkpoint_step,
    )
    keep = set(checkpoints[-keep_last:]) if keep_last > 0 else set()
    if best_checkpoint is not None:
        keep.add(best_checkpoint)

    for path in checkpoints:
        if path not in keep:
            path.unlink(missing_ok=True)


def save_checkpoint(
    path: Path,
    *,
    step: int,
    loss: float,
    best_loss: float | None,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "step": step,
            "loss": loss,
            "best_loss": best_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def main() -> int:
    args = parse_args()
    suite_root = Path.cwd()
    config = load_toml(resolve(args.config, suite_root))
    train_cfg = config["train"]
    paths_cfg = config["paths"]

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run from a CUDA-enabled environment or pass --device cpu.")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if train_cfg.get("dtype", "bf16") == "bf16" else torch.float16

    seed = int(train_cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    manifest_path = resolve(paths_cfg["manifest"], suite_root)
    dataset = ZImageLatentDataset(manifest_path, suite_root, limit=args.limit)
    sampler = BucketBatchSampler(dataset.rows, batch_size=int(train_cfg["batch_size"]), seed=seed)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_batch,
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )

    model = create_model(config).to(device=device, dtype=dtype)
    if train_cfg.get("gradient_checkpointing", False) and hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    output_dir = resolve(paths_cfg["output_dir"], suite_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    best_loss: float | None = None
    best_checkpoint: Path | None = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint.get("step", 0))
        best_loss = checkpoint.get("best_loss")
        if best_loss is not None:
            best_checkpoint = resolve(args.resume, suite_root)

    print(f"dataset={len(dataset)} manifest={manifest_path}")
    print(f"params={count_parameters(model):,}")
    print(f"device={device} dtype={dtype} {cuda_memory_summary(device)}")
    print(f"output_dir={output_dir}")

    data_iter = infinite_loader(loader)
    accum = int(train_cfg.get("grad_accum_steps", 1))
    max_steps = int(train_cfg["max_steps"])
    log_every = int(train_cfg.get("log_every", 10))
    save_every = int(train_cfg.get("save_every", 100))
    keep_last = int(train_cfg.get("keep_last_checkpoints", 0))
    keep_best = bool(train_cfg.get("keep_best_checkpoint", False))
    sigma_min = float(train_cfg.get("sigma_min", 0.0))
    sigma_max = float(train_cfg.get("sigma_max", 1.0))
    patch_size = int(config["model"]["patch_size"])
    f_patch_size = int(config["model"]["f_patch_size"])

    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_count = 0
    last = time.time()
    batches_per_epoch = len(loader)

    for step in range(start_step + 1, max_steps + 1):
        current_lr = learning_rate_for_step(step, train_cfg)
        set_optimizer_lr(optimizer, current_lr)
        step_loss = 0.0
        for _ in range(accum):
            batch = next(data_iter)
            x0 = batch["latents"].to(device=device, dtype=dtype)
            prompt_embeds = [emb.to(device=device, dtype=dtype) for emb in batch["prompt_embeds"]]
            noise = torch.randn_like(x0)
            sigma = torch.rand(x0.shape[0], device=device, dtype=torch.float32)
            sigma = sigma_min + (sigma_max - sigma_min) * sigma
            sigma_view = sigma.to(dtype).view(-1, 1, 1, 1)
            xt = (1.0 - sigma_view) * x0 + sigma_view * noise
            target = noise - x0
            model_time = model_time_from_sigma(sigma, train_cfg)

            xt_list = list(xt.unsqueeze(2).unbind(dim=0))
            raw_pred = model(
                xt_list,
                model_time,
                prompt_embeds,
                patch_size=patch_size,
                f_patch_size=f_patch_size,
                return_dict=False,
            )[0]
            pred = -torch.stack([item.squeeze(1) for item in raw_pred], dim=0).to(dtype)
            loss = F.mse_loss(pred.float(), target.float())
            (loss / accum).backward()
            step_loss += float(loss.detach())

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("max_grad_norm", 1.0)))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        loss_value = step_loss / accum
        running_loss += loss_value
        running_count += 1
        consumed_batches = step * accum
        epoch_idx = (consumed_batches - 1) // batches_per_epoch
        batch_idx = (consumed_batches - 1) % batches_per_epoch + 1

        if step % log_every == 0 or step == 1:
            now = time.time()
            avg = running_loss / max(running_count, 1)
            print(
                f"step={step} epoch={epoch_idx} batch={batch_idx}/{batches_per_epoch} "
                f"loss={loss_value:.6f} avg_loss={avg:.6f} "
                f"grad_norm={float(grad_norm):.4f} lr={current_lr:.6e} bucket={batch['bucket_id']} "
                f"{cuda_memory_summary(device)} dt={now - last:.2f}s"
            )
            running_loss = 0.0
            running_count = 0
            last = now

        if step % save_every == 0 or step == max_steps:
            ckpt_path = output_dir / f"checkpoint-{step:06d}.pt"
            is_best = best_loss is None or loss_value < best_loss
            if is_best:
                best_loss = loss_value
                best_checkpoint = ckpt_path
            save_checkpoint(
                ckpt_path,
                step=step,
                loss=loss_value,
                best_loss=best_loss,
                model=model,
                optimizer=optimizer,
                config=config,
            )
            if keep_last > 0 or keep_best:
                prune_checkpoints(output_dir, keep_last=keep_last, best_checkpoint=best_checkpoint if keep_best else None)
            best_text = f" best_loss={best_loss:.6f}" if best_loss is not None else ""
            print(f"saved {ckpt_path} loss={loss_value:.6f}{best_text}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
