[English](README.md) | [中文](README.zh-CN.md)

# nano-S3-DiT

`nano-S3-DiT` is a compact, pure-PyTorch S3DiT and flow-matching training project. It implements the diffusion transformer, training loop, and Euler sampler directly in this repository.

The repo includes a 52-sample EMNIST overfit dataset: one image for every `A-Z` and `a-z` class. Its Qwen caption embeddings and Z-Image VAE latents are already preprocessed under `cache/`, so training can start immediately after creating the environment.

## What is S3DiT?

S3DiT is a **Scalable Single-Stream Diffusion Transformer** architecture. Instead of processing text and image features in separate transformer branches, it projects text embeddings and image latent patches into one token sequence and lets them interact through shared self-attention blocks. The diffusion timestep is injected as adaptive conditioning.

This repository is a small educational implementation intended to make that core data flow easy to inspect, modify, and train.

## Architecture

The model architecture and training setup are primarily inspired by **Z-Image**, while deliberately keeping the implementation minimal:

- text tokens and image latent tokens are concatenated into a single stream;
- shared transformer blocks jointly process both modalities;
- timestep conditioning is applied through adaptive normalization;
- the model is trained with a flow-matching objective;
- Z-Image's VAE is used to encode and decode image latents.

This project is not an official Z-Image implementation or an exact reproduction of its full model.

**Z-Image links:** [GitHub](https://github.com/Tongyi-MAI/Z-Image) | [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image) | [Paper](https://arxiv.org/abs/2511.22699)

## Expected Result

After the overfit recipe runs successfully, inference should produce a result similar to the example below. The small image is the original 28x28 EMNIST training sample for the prompt `J`; the larger image is the 128x128 result generated from the checkpoint at step 8000 with seed 42.

| Original training sample | Generated result |
|:---:|:---:|
| <img src="assets/examples/emnist-j-reference.jpg" alt="Original EMNIST J training sample" width="112"> | <img src="assets/examples/emnist-j-step8000-seed42.png" alt="Generated J at step 8000 with seed 42" width="256"> |
| Prompt: `J`, 28x28 | Step 8000, seed 42, 128x128 |

## Contents

```text
nano_s3dit/models.py                         Pure PyTorch DiT
scripts/cache_text_embeddings.py             Cache Qwen caption embeddings
scripts/cache_vae_latents.py                 Cache Z-Image VAE latents
scripts/train.py                             Flow-matching training
scripts/infer.py                             Euler flow inference
configs/nano-s3dit-overfit-178m.toml         177,577,248 params (~178M), 2000 training steps
data/overfit-emnist-byclass-one-per-class    52 jpg/txt samples
cache/overfit-emnist-caption-embeds          Preprocessed Qwen embeddings
cache/overfit-emnist-vae-latents             Preprocessed Z-Image VAE latents
commands_setup.txt                           Environment and model download commands
commands_overfit.txt                         Data processing, training, and inference commands
```

## 1. Create The Conda Environment

```bash
conda create -n nano-s3dit python=3.12 -y
conda activate nano-s3dit
pip install -r requirements.txt
```

If the pinned CUDA/PyTorch wheel does not match your machine, install a compatible PyTorch build first, then install the remaining packages.

`hf` is the current Hugging Face CLI and is provided by the `huggingface_hub` Python package. This project lists that package explicitly in `requirements.txt`. Verify the installation with:

```bash
hf version
```

You can also install or upgrade it separately:

```bash
pip install -U huggingface_hub
```

Public models normally do not require authentication, but logging in provides higher request limits and is more reliable for large downloads. Authentication is also required for gated or private repositories:

```bash
hf auth login
```

## 2. Train Directly

The bundled cache contains everything needed by the training loop:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train.py --config configs/nano-s3dit-overfit-178m.toml --device cuda
```

The default recipe trains for 2000 steps and writes:

```text
checkpoints/nano-s3dit-overfit-178m/checkpoint-002000.pt
```

Training does not load Qwen3-0.6B or the Z-Image VAE.

## 3. Download Inference Models

Inference still needs Qwen3-0.6B to encode prompts and the Z-Image VAE to decode generated latents.

### Qwen3-0.6B Text Encoder

Caption embeddings for the bundled dataset are already cached. Download this model to encode new prompts during inference or to rebuild the cache after changing captions.

Target path:

```text
models/qwen3-0_6b/
```

Download:

```bash
mkdir -p models
hf download Qwen/Qwen3-0.6B --local-dir models/qwen3-0_6b
```

Manual download also works as long as the folder contains normal Hugging Face causal LM files such as `config.json`, `model.safetensors`, and tokenizer files.

### Z-Image VAE

Target path:

```text
models/z-image-vae/
```

Required files:

```text
models/z-image-vae/config.json
models/z-image-vae/diffusion_pytorch_model.safetensors
```

If the official Z-Image repo stores the VAE under a `vae/` subfolder, run:

```bash
mkdir -p models/z-image-download
hf download Tongyi-MAI/Z-Image-Turbo vae/config.json vae/diffusion_pytorch_model.safetensors --local-dir models/z-image-download
mkdir -p models/z-image-vae
cp models/z-image-download/vae/config.json models/z-image-vae/config.json
cp models/z-image-download/vae/diffusion_pytorch_model.safetensors models/z-image-vae/diffusion_pytorch_model.safetensors
```

If you already have these two files, place them directly under `models/z-image-vae/`.

## 4. Inference

Generate uppercase `A`:

```bash
python scripts/infer.py --config configs/nano-s3dit-overfit-178m.toml --checkpoint checkpoints/nano-s3dit-overfit-178m/checkpoint-002000.pt --prompt "A" --height 128 --width 128 --steps 50 --guidance-scale 0.0 --seed 42 --output outputs/overfit-A-step2000.png
```

## 5. Data Format

This repo already includes the overfit dataset:

```text
data/overfit-emnist-byclass-one-per-class/
  1.jpg
  1.txt
  2.jpg
  2.txt
  ...
  52.jpg
  52.txt
```

Every image needs a same-stem `.txt` caption. Use the same format for your own data.

## 6. Optional: Rebuild Caption Embeddings

Run this only after changing the dataset, captions, or text encoder:

```bash
python scripts/cache_text_embeddings.py --data-dir data/overfit-emnist-byclass-one-per-class --model models/qwen3-0_6b --model-layout causal-lm --quantization none --dtype bf16 --batch-size 52 --output-dir cache/overfit-emnist-caption-embeds/text-embeds-qwen3-0_6b-bf16
```

Output:

```text
cache/overfit-emnist-caption-embeds/text-embeds-qwen3-0_6b-bf16/
  manifest.jsonl
  embeddings/*.safetensors
```

## 7. Optional: Rebuild Z-Image VAE Latents

Run this only after changing the images, VAE, or preprocessing resolution:

```bash
python scripts/cache_vae_latents.py --caption-embeds-dir cache/overfit-emnist-caption-embeds/text-embeds-qwen3-0_6b-bf16 --vae models/z-image-vae --output-dir cache/overfit-emnist-vae-latents/z-image-vae-128-bf16 --bucket-max-pixels 16384 --bucket-min-side 128 --bucket-max-side 128 --bucket-step 16 --batch-size 52 --dtype bf16 --device cuda
```

Output:

```text
cache/overfit-emnist-vae-latents/z-image-vae-128-bf16/
  manifest.jsonl
  latents/*.safetensors
```

## 8. Objective

Training uses flow matching:

```python
x_t = (1 - sigma) * x0 + sigma * noise
target = noise - x0
model_time = 1 - sigma
```

`sigma` is always the noise strength:

```text
sigma = 1   pure noise
sigma = 0   clean latent
```

`model_time = 1 - sigma` means the model sees time from noisy to clean:

```text
inference starts: model_time = 0
inference ends:   model_time = 1
```

This recipe does not train an unconditional branch, so use `guidance_scale = 0.0`.
