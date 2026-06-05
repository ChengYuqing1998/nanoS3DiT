from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class NanoS3DiTConfig:
    in_channels: int = 16
    patch_size: int = 2
    dim: int = 768
    n_layers: int = 18
    n_heads: int = 12
    cap_feat_dim: int = 1024
    mlp_ratio: float = 4.0
    time_embed_dim: int = 256


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimeEmbedder(nn.Module):
    def __init__(self, out_dim: int, freq_dim: int = 256, hidden_dim: int = 1024):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = timestep_embedding(t, self.freq_dim)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class DiTBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True, bias=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, dim),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada(c).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        h = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)[0]
        x = x + gate_msa[:, None] * h
        h = self.norm2(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        x = x + gate_mlp[:, None] * self.mlp(h)
        return x


class NanoS3DiT(nn.Module):
    def __init__(self, config: NanoS3DiTConfig):
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels
        patch_dim = config.in_channels * config.patch_size * config.patch_size
        self.patch_in = nn.Linear(patch_dim, config.dim)
        self.patch_out = nn.Linear(config.dim, patch_dim)
        self.time_embed = TimeEmbedder(config.dim, freq_dim=config.time_embed_dim)
        self.cap_embed = nn.Sequential(
            nn.LayerNorm(config.cap_feat_dim, eps=1e-6),
            nn.Linear(config.cap_feat_dim, config.dim),
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(config.dim, config.n_heads, config.mlp_ratio) for _ in range(config.n_layers)]
        )
        self.final_norm = nn.LayerNorm(config.dim, elementwise_affine=False, eps=1e-6)
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def _patchify(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        b, c, h, w = x.shape
        p = self.config.patch_size
        if h % p != 0 or w % p != 0:
            raise ValueError(f"Latent height/width must be divisible by patch_size={p}, got {(h, w)}")
        x = x.reshape(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(b, (h // p) * (w // p), c * p * p)
        return x, (h // p, w // p)

    def _unpatchify(self, x: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        b, n, d = x.shape
        gh, gw = grid
        p = self.config.patch_size
        c = self.config.in_channels
        if n != gh * gw:
            raise ValueError(f"Token count {n} does not match grid {grid}")
        x = x.reshape(b, gh, gw, c, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(b, c, gh * p, gw * p)
        return x

    def _image_pos(self, grid: tuple[int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        gh, gw = grid
        ys = torch.linspace(-1.0, 1.0, gh, device=device)
        xs = torch.linspace(-1.0, 1.0, gw, device=device)
        y, x = torch.meshgrid(ys, xs, indexing="ij")
        pos = torch.stack([x, y], dim=-1).reshape(1, gh * gw, 2)
        repeats = math.ceil(self.config.dim / 4)
        freqs = torch.arange(1, repeats + 1, device=device, dtype=torch.float32)[None, None, :]
        x = pos[..., :1] * freqs * math.pi
        y = pos[..., 1:] * freqs * math.pi
        emb = torch.cat([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=-1)
        return emb[..., : self.config.dim].to(dtype)

    def forward(
        self,
        x_list: list[torch.Tensor],
        t: torch.Tensor,
        prompt_embeds: list[torch.Tensor],
        patch_size: int | None = None,
        f_patch_size: int | None = None,
        return_dict: bool = False,
    ):
        del f_patch_size
        if patch_size is not None and int(patch_size) != self.config.patch_size:
            raise ValueError(f"This model was created with patch_size={self.config.patch_size}, got {patch_size}")

        x = torch.stack([item.squeeze(1) if item.ndim == 4 else item for item in x_list], dim=0)
        img_tokens, grid = self._patchify(x)
        img_tokens = self.patch_in(img_tokens)
        img_tokens = img_tokens + self._image_pos(grid, img_tokens.device, img_tokens.dtype)

        text_lens = [emb.shape[0] for emb in prompt_embeds]
        max_text = max(text_lens)
        text = img_tokens.new_zeros((len(prompt_embeds), max_text, self.config.cap_feat_dim))
        text_mask = torch.ones((len(prompt_embeds), max_text), device=img_tokens.device, dtype=torch.bool)
        for i, emb in enumerate(prompt_embeds):
            n = emb.shape[0]
            text[i, :n] = emb.to(device=img_tokens.device, dtype=text.dtype)
            text_mask[i, :n] = False
        text_tokens = self.cap_embed(text)

        tokens = torch.cat([text_tokens, img_tokens], dim=1)
        img_mask = torch.zeros((tokens.shape[0], img_tokens.shape[1]), device=tokens.device, dtype=torch.bool)
        key_padding_mask = torch.cat([text_mask, img_mask], dim=1)
        cond = self.time_embed(t.to(tokens.device)).to(tokens.dtype)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                tokens = checkpoint(block, tokens, cond, key_padding_mask, use_reentrant=False)
            else:
                tokens = block(tokens, cond, key_padding_mask)

        img_tokens = tokens[:, -img_tokens.shape[1] :]
        img_tokens = self.final_norm(img_tokens)
        out = self._unpatchify(self.patch_out(img_tokens), grid)
        out_list = [item.unsqueeze(1) for item in out.unbind(dim=0)]
        return (out_list,) if not return_dict else {"sample": out_list}


def build_model_from_config(model_cfg: dict) -> NanoS3DiT:
    cfg = NanoS3DiTConfig(
        in_channels=int(model_cfg["in_channels"]),
        patch_size=int(model_cfg["patch_size"]),
        dim=int(model_cfg["dim"]),
        n_layers=int(model_cfg["n_layers"]),
        n_heads=int(model_cfg["n_heads"]),
        cap_feat_dim=int(model_cfg["cap_feat_dim"]),
        mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
        time_embed_dim=int(model_cfg.get("time_embed_dim", 256)),
    )
    return NanoS3DiT(cfg)
