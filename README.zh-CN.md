# nano-s3dit

`nano-s3dit` 是一个极简的 S3DiT / Z-Image 风格 flow matching 训练项目。DiT 模型、训练循环和推理 sampler 都在本仓库内实现，不依赖 Flow-Factory。

仓库内置一个 52 张图的 EMNIST overfit 数据集，每个 `A-Z` 和 `a-z` 类别各 1 张，用来验证完整链路。对应的 Qwen caption embeddings 和 Z-Image VAE latents 已经预处理并提交到 `cache/`，创建环境后可以直接运行训练，不需要先下载模型或执行缓存脚本。

## 项目内容

```text
nano_s3dit/models.py                         纯 PyTorch DiT
scripts/cache_text_embeddings.py             缓存 Qwen caption embedding
scripts/cache_vae_latents.py                 缓存 Z-Image VAE latent
scripts/train.py                             flow matching 训练
scripts/infer.py                             Euler flow 推理
configs/nano-s3dit-overfit-176m.toml         约 178M 参数，训练 2000 step
data/overfit-emnist-byclass-one-per-class    52 个 jpg/txt 样本
cache/overfit-emnist-caption-embeds          已预处理的 Qwen embeddings
cache/overfit-emnist-vae-latents             已预处理的 Z-Image VAE latents
commands_setup.txt                           环境和模型下载命令
commands_overfit.txt                         数据处理、训练、推理命令
```

## 1. 创建 Conda 环境

```bash
conda create -n nano-s3dit python=3.12 -y
conda activate nano-s3dit
pip install -r requirements.txt
```

如果你的 CUDA/PyTorch wheel 与 `requirements.txt` 不匹配，先按你的机器安装可用的 PyTorch，再安装其余包。

`hf` 是 Hugging Face 当前的命令行工具，由 `huggingface_hub` Python 包提供。本项目已经在 `requirements.txt` 中显式包含该包。安装后可以验证：

```bash
hf version
```

也可以单独安装或升级：

```bash
pip install -U huggingface_hub
```

公开模型通常无需登录，但登录后请求限额更高，下载大型文件时更稳定；访问 gated/private 仓库也必须登录：

```bash
hf auth login
```

## 2. 直接训练

仓库内置的缓存已经包含训练所需的 caption embeddings 和 VAE latents，因此可以直接执行：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train.py --config configs/nano-s3dit-overfit-176m.toml --device cuda
```

默认训练 2000 step，checkpoint 输出到：

```text
checkpoints/nano-s3dit-overfit-176m/checkpoint-002000.pt
```

训练阶段不会加载 Qwen3-0.6B 或 Z-Image VAE。

## 3. 下载推理模型

推理时仍然需要 Qwen3-0.6B 编码 prompt，并使用 Z-Image VAE 解码生成的 latent。

### Qwen3-0.6B Text Encoder

内置数据的 caption embedding 已经预缓存。下载这个模型是为了在推理时编码新的 prompt，或者在替换 caption 后重新生成缓存。

目标路径：

```text
models/qwen3-0_6b/
```

下载命令：

```bash
mkdir -p models
hf download Qwen/Qwen3-0.6B --local-dir models/qwen3-0_6b
```

也可以手动下载，只要目录里有 `config.json`、`model.safetensors`、`tokenizer.json` 等 Hugging Face causal LM 文件即可。

### Z-Image VAE

目标路径：

```text
models/z-image-vae/
```

目录内至少需要：

```text
models/z-image-vae/config.json
models/z-image-vae/diffusion_pytorch_model.safetensors
```

如果你从官方 Z-Image 仓库下载，并且 VAE 在 `vae/` 子目录，可以用：

```bash
mkdir -p models/z-image-download
hf download Tongyi-MAI/Z-Image-Turbo vae/config.json vae/diffusion_pytorch_model.safetensors --local-dir models/z-image-download
mkdir -p models/z-image-vae
cp models/z-image-download/vae/config.json models/z-image-vae/config.json
cp models/z-image-download/vae/diffusion_pytorch_model.safetensors models/z-image-vae/diffusion_pytorch_model.safetensors
```

如果你已经有这两个文件，直接放到 `models/z-image-vae/` 即可。

## 4. 推理

生成大写 `A`：

```bash
python scripts/infer.py --config configs/nano-s3dit-overfit-176m.toml --checkpoint checkpoints/nano-s3dit-overfit-176m/checkpoint-002000.pt --prompt "A" --height 128 --width 128 --steps 50 --guidance-scale 0.0 --seed 42 --output outputs/overfit-A-step2000.png
```

## 5. 数据格式

本仓库已经带了 overfit 数据：

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

每张图片需要一个同名 `.txt` caption 文件。你换自己的数据时也用这个格式。

## 6. 可选：重新缓存 Caption Embeddings

只有替换数据、caption 或 text encoder 时才需要执行：

```bash
python scripts/cache_text_embeddings.py --data-dir data/overfit-emnist-byclass-one-per-class --model models/qwen3-0_6b --model-layout causal-lm --quantization none --dtype bf16 --batch-size 52 --output-dir cache/overfit-emnist-caption-embeds/text-embeds-qwen3-0_6b-bf16
```

输出：

```text
cache/overfit-emnist-caption-embeds/text-embeds-qwen3-0_6b-bf16/
  manifest.jsonl
  embeddings/*.safetensors
```

## 7. 可选：重新缓存 Z-Image VAE Latents

只有替换图片、VAE 或预处理尺寸时才需要执行：

```bash
python scripts/cache_vae_latents.py --caption-embeds-dir cache/overfit-emnist-caption-embeds/text-embeds-qwen3-0_6b-bf16 --vae models/z-image-vae --output-dir cache/overfit-emnist-vae-latents/z-image-vae-128-bf16 --bucket-max-pixels 16384 --bucket-min-side 128 --bucket-max-side 128 --bucket-step 16 --batch-size 52 --dtype bf16 --device cuda
```

输出：

```text
cache/overfit-emnist-vae-latents/z-image-vae-128-bf16/
  manifest.jsonl
  latents/*.safetensors
```

## 8. 训练目标

训练使用 flow matching：

```python
x_t = (1 - sigma) * x0 + sigma * noise
target = noise - x0
model_time = 1 - sigma
```

`sigma` 永远表示噪声强度：

```text
sigma = 1   纯噪声
sigma = 0   干净 latent
```

`model_time = 1 - sigma` 表示模型看到的时间从噪声端到干净端：

```text
推理开始: model_time = 0
推理结束: model_time = 1
```

当前 recipe 没有训练 unconditional 分支，所以 `guidance_scale` 固定用 `0.0`。
