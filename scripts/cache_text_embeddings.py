#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache Z-Image/Qwen text embeddings from numbered .txt captions in a dataset directory."
    )
    parser.add_argument(
        "--data-dir",
        default="data/overfit-emnist-byclass-one-per-class",
        help="Extracted dataset directory containing files such as 1.txt and 1.jpg.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root used to write project-relative paths into manifest.jsonl.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="HF repo or local model path. Use Qwen/Qwen3-0.6B for a small custom pretraining text encoder.",
    )
    parser.add_argument(
        "--model-layout",
        choices=["causal-lm", "z-image"],
        default="causal-lm",
        help="causal-lm loads tokenizer/model directly; z-image loads tokenizer/ and text_encoder/ subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        default="cache/overfit-emnist-caption-embeds/text-embeds-qwen3-0_6b-bf16",
        help="Directory for per-sample safetensors and manifest.jsonl.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--quantization",
        choices=["none", "4bit"],
        default="none",
        help="Use none for normal bf16/fp16 loading, or 4bit for bitsandbytes NF4 loading.",
    )
    parser.add_argument(
        "--include-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Qwen chat-template prompt encoding with enable_thinking=True when supported.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap for smoke testing.",
    )
    return parser.parse_args()


def project_relative(path: Path, project_root: Path) -> str:
    path = path.resolve()
    project_root = project_root.resolve()
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def numbered_txt_paths_from_dir(data_dir: Path) -> list[Path]:
    paths = [path for path in data_dir.glob("*.txt") if path.is_file()]

    def key(path: Path) -> tuple[int, str]:
        stem = path.stem
        return (int(stem), path.name) if stem.isdigit() else (10**12, path.name)

    return sorted(paths, key=key)


def format_prompt(tokenizer: AutoTokenizer, caption: str, include_thinking: bool) -> str:
    messages = [{"role": "user", "content": caption}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=include_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def iter_batches(items: list[dict], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    embed_dir = output_dir / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)

    compute_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    quant_config = None
    if args.quantization == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer_kwargs = {"trust_remote_code": True}
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "dtype": compute_dtype,
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
    if args.model_layout == "z-image":
        tokenizer_kwargs["subfolder"] = "tokenizer"
        model_kwargs["subfolder"] = "text_encoder"

    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
    text_encoder = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    text_encoder.eval()

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    caption_sources = [
        {
            "sample_id": path.stem,
            "caption_name": path.name,
            "caption_path": path,
            "image_path": data_dir / f"{path.stem}.jpg",
        }
        for path in numbered_txt_paths_from_dir(data_dir)
    ]

    if args.limit is not None:
        caption_sources = caption_sources[: args.limit]

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for start_idx, batch_sources in iter_batches(caption_sources, args.batch_size):
            captions = []
            prompts = []
            for source in batch_sources:
                txt_name = source["caption_name"]
                caption_path = source["caption_path"]
                image_path = source["image_path"]
                if image_path is not None and not image_path.exists():
                    raise FileNotFoundError(f"Missing paired image for {txt_name}: {image_path}")
                caption = caption_path.read_text(encoding="utf-8").strip()
                captions.append(caption)
                prompts.append(format_prompt(tokenizer, caption, args.include_thinking))

            inputs = tokenizer(
                prompts,
                padding="max_length",
                max_length=args.max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = inputs.input_ids.to(text_encoder.device)
            attention_mask = inputs.attention_mask.to(text_encoder.device).bool()

            with torch.inference_mode():
                outputs = text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                full_embeds = outputs.hidden_states[-2]

            for offset, source in enumerate(batch_sources):
                sample_id = source["sample_id"]
                txt_name = source["caption_name"]
                caption_path = source["caption_path"]
                image_path = source["image_path"]
                caption = captions[offset]
                sample_mask = attention_mask[offset]
                valid_embeds = full_embeds[offset][sample_mask].to(compute_dtype).cpu()
                sample_input_ids = input_ids[offset].cpu().to(torch.int64)
                sample_attention_mask = sample_mask.cpu().to(torch.int64)

                cache_file = embed_dir / f"{sample_id}.safetensors"
                save_file(
                    {
                        "prompt_embeds": valid_embeds.contiguous(),
                        "input_ids": sample_input_ids,
                        "attention_mask": sample_attention_mask,
                        "valid_length": torch.tensor([valid_embeds.shape[0]], dtype=torch.int64),
                    },
                    str(cache_file),
                )

                record = {
                    "id": sample_id,
                    "caption_file": txt_name,
                    "image_file": f"{sample_id}.jpg",
                    "embedding_file": str(cache_file.relative_to(output_dir)),
                    "project_relative_caption_path": (
                        project_relative(caption_path, project_root) if caption_path is not None else None
                    ),
                    "project_relative_image_path": (
                        project_relative(image_path, project_root) if image_path is not None else None
                    ),
                    "project_relative_embedding_path": project_relative(cache_file, project_root),
                    "caption": caption,
                    "valid_length": int(valid_embeds.shape[0]),
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")

            done = min(start_idx + len(batch_sources), len(caption_sources))
            print(f"[{done}/{len(caption_sources)}] cached batch -> {embed_dir}")

    meta = {
        "data_dir": str(data_dir),
        "project_relative_data_dir": project_relative(data_dir, project_root),
        "model": args.model,
        "model_layout": args.model_layout,
        "quantization": "none" if args.quantization == "none" else "bitsandbytes-4bit-nf4-double-quant",
        "compute_dtype": args.dtype,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "hidden_state_layer": -2,
        "include_thinking": args.include_thinking,
        "num_samples": len(caption_sources),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Done. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
