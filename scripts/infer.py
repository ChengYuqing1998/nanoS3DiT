#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from train import create_model, load_toml, model_time_from_sigma, resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a checkpoint from the Z-Image-like pretraining run.")
    parser.add_argument(
        "--config",
        default="configs/nano-s3dit-overfit-176m.toml",
        help="Training config used to define the transformer shape and source paths.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a saved training checkpoint, for example checkpoint-001000.pt.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt to generate from.",
    )
    parser.add_argument(
        "--negative-prompt",
        default="",
        help="Optional negative prompt. Only used when guidance_scale > 0.",
    )
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vae", default="models/z-image-vae")
    parser.add_argument("--text-encoder", default="models/qwen3-0_6b")
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=512,
        help="Maximum caption token length for the text encoder.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to outputs/<checkpoint>.png",
    )
    return parser.parse_args()


def infer_output_path(checkpoint: Path, output: str | None, suite_root: Path) -> Path:
    if output is not None:
        return resolve(output, suite_root)
    stem = checkpoint.stem
    return resolve(f"outputs/{stem}.png", suite_root)


def load_components(
    config: dict,
    checkpoint_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    vae_path: Path,
    text_encoder_path: Path,
):
    suite_root = Path.cwd()

    from diffusers import AutoencoderKL
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    effective_config = config
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
        effective_config = checkpoint["config"]
    model = create_model(effective_config)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device=device, dtype=dtype)
    model.eval()

    vae = AutoencoderKL.from_pretrained(str(vae_path), torch_dtype=dtype)
    vae.to(device=device)
    vae.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(text_encoder_path), trust_remote_code=True)
    text_encoder = AutoModelForCausalLM.from_pretrained(
        str(text_encoder_path),
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    text_encoder.to(device=device)
    text_encoder.eval()
    return model, vae, tokenizer, text_encoder, effective_config


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().float().cpu().clamp(0.0, 1.0)
    image = (image * 255.0).round().to(torch.uint8)
    image = image.permute(1, 2, 0).numpy()
    return Image.fromarray(image)


def encode_prompt(tokenizer, text_encoder, prompt: str, device: torch.device, max_sequence_length: int) -> list[torch.Tensor]:
    messages = [{"role": "user", "content": prompt}]
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(
        [prompt_text],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device).bool()
    with torch.inference_mode():
        hidden = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        ).hidden_states[-2]
    return [hidden[0][attention_mask[0]]]


@torch.inference_mode()
def sample_with_training_time_convention(
    model,
    vae,
    tokenizer,
    text_encoder,
    prompt: str,
    train_cfg: dict,
    *,
    height: int,
    width: int,
    steps: int,
    generator: torch.Generator,
    max_sequence_length: int,
) -> Image.Image:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    prompt_embeds = encode_prompt(tokenizer, text_encoder, prompt, device=device, max_sequence_length=max_sequence_length)

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    latent_h = 2 * (int(height) // (vae_scale_factor * 2))
    latent_w = 2 * (int(width) // (vae_scale_factor * 2))
    latents = torch.randn(
        (1, model.in_channels, latent_h, latent_w),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )

    sigmas = torch.linspace(1.0, 0.0, int(steps) + 1, device=device, dtype=torch.float32)
    patch_size = int(model.config.patch_size)
    for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:]):
        sigma_batch = sigma.expand(latents.shape[0]).to(torch.float32)
        model_time = model_time_from_sigma(sigma_batch, train_cfg)
        latent_model_input = latents.to(dtype).unsqueeze(2)
        model_out = model(
            list(latent_model_input.unbind(dim=0)),
            model_time,
            prompt_embeds,
            patch_size=patch_size,
            f_patch_size=1,
            return_dict=False,
        )[0]
        velocity = -torch.stack([item.float() for item in model_out], dim=0).squeeze(2)
        latents = latents + velocity * (sigma_next - sigma)

    decode_latents = (latents.to(vae.dtype) / vae.config.scaling_factor) + vae.config.shift_factor
    image = vae.decode(decode_latents, return_dict=False)[0][0]
    image = (image / 2.0 + 0.5).clamp(0.0, 1.0)
    return tensor_to_pil(image)


def main() -> int:
    args = parse_args()
    suite_root = Path.cwd()
    config = load_toml(resolve(args.config, suite_root))
    checkpoint_path = resolve(args.checkpoint, suite_root)
    output_path = infer_output_path(checkpoint_path, args.output, suite_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run in a CUDA-enabled environment or pass --device cpu.")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if config["train"].get("dtype", "bf16") == "bf16" else torch.float16

    seed = int(args.seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    model, vae, tokenizer, text_encoder, effective_config = load_components(
        config,
        checkpoint_path,
        device=device,
        dtype=dtype,
        vae_path=resolve(args.vae, suite_root),
        text_encoder_path=resolve(args.text_encoder, suite_root),
    )

    generator = torch.Generator(device=device).manual_seed(seed)
    if float(args.guidance_scale) != 0.0:
        raise ValueError("This custom pretraining sampler currently expects --guidance-scale 0.0.")

    image = sample_with_training_time_convention(
        model,
        vae,
        tokenizer,
        text_encoder,
        args.prompt,
        effective_config["train"],
        height=int(args.height),
        width=int(args.width),
        steps=int(args.steps),
        generator=generator,
        max_sequence_length=int(args.max_sequence_length),
    )

    image.save(output_path)
    print(f"saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
