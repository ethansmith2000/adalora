"""
Vision Transformer (ViT) implementation for ImageNet training.
Inspired by the transformer.py architecture with clean, minimal code.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from conditioning import ConditionedLinear, VARIANTS, project
from vit_initialization import MLP_INITIALIZERS, initialize_mlp


class PatchEmbed(nn.Module):
    """Convert image to patch embeddings."""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class Attention(nn.Module):
    """Multi-head self-attention with optional QK normalization."""
    def __init__(self, dim, heads=8, qk_norm=True):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qk_norm = qk_norm
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
    
    def forward(self, x, condition=None):
        B, N, C = x.shape
        qkv = project(self.to_qkv, x, condition).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, heads, N, head_dim)
        
        if self.qk_norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        
        # Use flash attention via SDPA
        attn_out = F.scaled_dot_product_attention(q, k, v)
        
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        return self.to_out(attn_out)


class MLP(nn.Module):
    """MLP with GELU activation."""
    def __init__(self, dim, hidden_dim=None, drop=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x, condition=None):
        x = project(self.fc1, x, condition)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture."""
    def __init__(self, dim, heads, mlp_ratio=4.0, qk_norm=True, drop=0.0,
                 detach_condition=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, qk_norm=qk_norm)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)
        self.detach_condition = detach_condition
    
    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h.detach() if self.detach_condition else h)
        h = self.norm2(x)
        x = x + self.mlp(h, h.detach() if self.detach_condition else h)
        return x


class ViT(nn.Module):
    """
    Vision Transformer for image classification.
    
    Configurations:
    - ViT-Ti: dim=192, depth=12, heads=3   (~5.7M params)
    - ViT-S:  dim=384, depth=12, heads=6   (~22M params)
    - ViT-B:  dim=768, depth=12, heads=12  (~86M params)
    - ViT-L:  dim=1024, depth=24, heads=16 (~307M params)
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        heads=12,
        mlp_ratio=4.0,
        qk_norm=True,
        drop_rate=0.0,
        gradient_checkpointing=False,
        conditioning="plain",
        conditioning_placement="mlp_in",
        condition_rank=2,
        condition_n_basis=16,
        condition_chunk=2048,
        detach_condition=False,
        condition_gain=1.0,
        condition_factor_offsets=False,
        mlp_init="legacy",
        condition_normalization="none",
        condition_gradient_scale=1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.gradient_checkpointing = gradient_checkpointing
        if mlp_init not in MLP_INITIALIZERS:
            raise ValueError('Invalid MLP initializer')
        self.mlp_init = mlp_init
        if conditioning not in VARIANTS:
            raise ValueError(f"conditioning must be one of {VARIANTS}")
        if conditioning_placement not in ("mlp_in", "qkv", "both"):
            raise ValueError("conditioning_placement must be mlp_in, qkv or both")
        if img_size % patch_size or embed_dim % heads:
            raise ValueError("image size must divide into patches and width into heads")
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, heads, mlp_ratio, qk_norm, drop_rate, detach_condition)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        if mlp_init != "legacy":
            for block in self.blocks:
                initialize_mlp(block.mlp, embed_dim, mlp_init)
        # Attach adapters after base initialization so generic Linear init cannot
        # overwrite zero scale/shift heads. The base is identical across variants
        # built with the same seed, including the classifier and patch embedding.
        if conditioning != "plain":
            for block in self.blocks:
                for module, attr in ((block.mlp, "fc1"), (block.attn, "to_qkv")):
                    if ((attr == "fc1" and conditioning_placement in ("mlp_in", "both"))
                            or (attr == "to_qkv" and conditioning_placement in ("qkv", "both"))):
                        setattr(module, attr, ConditionedLinear(
                            getattr(module, attr), conditioning, condition_rank,
                            condition_n_basis, condition_chunk, condition_gain,
                            condition_factor_offsets, condition_normalization,
                            condition_gradient_scale))
    
    def _init_weights(self):
        # Position embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Zero-init the last layer for better training dynamics
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x, targets=None):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Prepend class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        if self.gradient_checkpointing and self.training:
            for block in self.blocks:
                x = checkpoint(
                    block, x, use_reentrant=False
                )
        else:
            for block in self.blocks:
                x = block(x)
        
        # Classification head (use CLS token)
        x = self.norm(x)
        cls_out = x[:, 0]
        logits = self.head(cls_out)
        
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
            return loss, logits
        return logits


def vit_tiny(num_classes=1000, **kwargs):
    return ViT(embed_dim=192, depth=12, heads=3, num_classes=num_classes, **kwargs)

def vit_small(num_classes=1000, **kwargs):
    return ViT(embed_dim=384, depth=12, heads=6, num_classes=num_classes, **kwargs)

def vit_base(num_classes=1000, **kwargs):
    return ViT(embed_dim=768, depth=12, heads=12, num_classes=num_classes, **kwargs)

def vit_large(num_classes=1000, **kwargs):
    return ViT(embed_dim=1024, depth=24, heads=16, num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    # Quick test
    model = vit_small(num_classes=1000)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"ViT-S output shape: {y.shape}")
    print(f"ViT-S params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    model = vit_base(num_classes=1000)
    y = model(x)
    print(f"ViT-B output shape: {y.shape}")
    print(f"ViT-B params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
