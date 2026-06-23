"""
BhuViveka-VLM — Self-contained model definitions (inference only)
==================================================================
Every neural-network class needed to run the two-stage GLOF pipeline is
inlined here so this folder has NO dependency on the training code base.

  Stage 1  Prithvi-EO-2.0-300M (LoRA) + SpectralAdapter + LakeDecoder  -> lake mask
  Stage 2  ReasoningHead (visual CLS + physics)                        -> risk class

The Stage-1 checkpoint (checkpoints/seg_300m_best.pt) already embeds the full
backbone weights (frozen base + LoRA), so the original 1.3 GB Prithvi file is
NOT required here — we build the architecture with random init and let
load_state_dict() fill in every weight.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── geometry constants (must match training) ──────────────────
PATCH_SIZE_INPUT = 128     # raw patch resolution
PATCH_SIZE_MODEL = 448     # encoder input for the segmentation path (grid 28x28)
LORA_RANK  = 16
LORA_ALPHA = 32

PRITHVI_300M = {
    "patch_size": 16, "img_size": 448, "in_chans": 6, "num_frames": 1,
    "embed_dim": 1024, "depth": 24, "num_heads": 16, "mlp_ratio": 4.0,
}


# ============================================================
# PRITHVI-EO-2.0 ViT ENCODER
# ============================================================
class PatchEmbed3D(nn.Module):
    def __init__(self, img_size=448, patch_size=16, num_frames=1, in_chans=6, embed_dim=1024):
        super().__init__()
        self.num_patches_per_frame = (img_size // patch_size) ** 2
        self.num_patches = num_frames * self.num_patches_per_frame
        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=(1, patch_size, patch_size),
                              stride=(1, patch_size, patch_size))

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=16, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def _sincos_1d(embed_dim, pos):
    omega = torch.arange(embed_dim // 2, dtype=torch.float32) / (embed_dim / 2.0)
    omega = 1.0 / (10000 ** omega)
    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=True):
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0).reshape(2, 1, grid_size, grid_size)
    emb_h = _sincos_1d(embed_dim // 2, grid[0])
    emb_w = _sincos_1d(embed_dim // 2, grid[1])
    pos_embed = torch.cat([emb_h, emb_w], dim=1).reshape(grid_size * grid_size, embed_dim)
    if cls_token:
        pos_embed = torch.cat([torch.zeros(1, embed_dim), pos_embed], dim=0)
    return pos_embed.unsqueeze(0)


class PrithviViTEncoder(nn.Module):
    def __init__(self, cfg=PRITHVI_300M):
        super().__init__()
        self.embed_dim = cfg["embed_dim"]
        self.patch_size = cfg["patch_size"]
        self.img_size = cfg["img_size"]
        self.grid_size = self.img_size // self.patch_size      # 28
        self.patch_embed = PatchEmbed3D(cfg["img_size"], cfg["patch_size"],
                                        cfg["num_frames"], cfg["in_chans"], cfg["embed_dim"])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg["embed_dim"]))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, cfg["embed_dim"]),
                                      requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(cfg["embed_dim"], cfg["num_heads"], cfg["mlp_ratio"]) for _ in range(cfg["depth"])])
        self.norm = nn.LayerNorm(cfg["embed_dim"], eps=1e-6)
        with torch.no_grad():
            self.pos_embed.copy_(get_2d_sincos_pos_embed(self.embed_dim, self.grid_size, True))
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(2)                    # add time dim -> (B,6,1,H,W)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :x.size(1)]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# ============================================================
# STAGE 1 — SEGMENTATION
# ============================================================
class SpectralAdapter(nn.Module):
    def __init__(self, n_bands=6, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_bands, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, n_bands, 1),
        )
        self.bn = nn.BatchNorm2d(n_bands)

    def forward(self, x):
        return self.bn(x + self.net(x))


class LakeDecoder(nn.Module):
    def __init__(self, embed_dim=1024, grid_size=28, out_size=128):
        super().__init__()
        self.grid_size = grid_size
        self.out_size = out_size
        self.decode = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=1), nn.GELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, tokens):
        patch_tokens = tokens[:, 1:, :]                 # drop CLS
        B, N, D = patch_tokens.shape
        feat = patch_tokens.permute(0, 2, 1).reshape(B, D, self.grid_size, self.grid_size)
        out = self.decode(feat)
        out = F.interpolate(out, size=(self.out_size, self.out_size),
                            mode="bilinear", align_corners=False)
        return out                                       # (B,1,128,128)


class BhuVivekaSeg(nn.Module):
    def __init__(self, backbone, embed_dim):
        super().__init__()
        self.adapter = SpectralAdapter(6)
        self.backbone = backbone
        self.decoder = LakeDecoder(embed_dim=embed_dim,
                                   grid_size=PATCH_SIZE_MODEL // 16,
                                   out_size=PATCH_SIZE_INPUT)

    def forward(self, x):
        x = self.adapter(x)
        x = F.interpolate(x, size=(PATCH_SIZE_MODEL, PATCH_SIZE_MODEL),
                          mode="bilinear", align_corners=False)
        tokens = self.backbone(x)
        return self.decoder(tokens)


# ============================================================
# STAGE 2 — RISK REASONING HEAD
# ============================================================
class ReasoningHead(nn.Module):
    def __init__(self, visual_dim=1024, physics_dim=3, context_dim=256):
        super().__init__()
        self.visual_proj = nn.Sequential(
            nn.LayerNorm(visual_dim), nn.Linear(visual_dim, context_dim), nn.GELU())
        self.physics_proj = nn.Sequential(
            nn.Linear(physics_dim, 32), nn.GELU(), nn.Linear(32, context_dim))
        self.fuse = nn.Sequential(
            nn.Linear(context_dim * 2, context_dim), nn.GELU(), nn.LayerNorm(context_dim))
        self.risk_head = nn.Linear(context_dim, 3)

    def forward(self, visual_feat, physics):
        vf = self.visual_proj(visual_feat)
        pf = self.physics_proj(physics)
        ctx = self.fuse(torch.cat([vf, pf], dim=-1))
        return self.risk_head(ctx), ctx


# ============================================================
# BUILDERS
# ============================================================
def build_lora_backbone():
    """Random-init Prithvi-300M wrapped with the same LoRA config used in training.
    Weights are filled later by the Stage-1 checkpoint."""
    from peft import LoraConfig, get_peft_model
    enc = PrithviViTEncoder(PRITHVI_300M)
    for p in enc.parameters():
        p.requires_grad = False
    targets = [n for n, m in enc.named_modules()
               if isinstance(m, nn.Linear) and n.endswith(".qkv")]
    if not targets:
        targets = [n for n, m in enc.named_modules()
                   if isinstance(m, nn.Linear) and "attn" in n.lower()]
    cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=targets,
                     lora_dropout=0.05, bias="none", modules_to_save=["patch_embed"])
    return get_peft_model(enc, cfg), enc.embed_dim
