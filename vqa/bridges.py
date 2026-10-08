"""Trainable bridges that turn frozen vision features into visual tokens for the language model."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """Self-attention over the queries, cross-attention to the image, feed-forward. Pre-norm."""

    def __init__(self, width, heads, dropout):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(width), nn.LayerNorm(width), nn.LayerNorm(width)
        self.self_attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * width, width))
        self.drop = nn.Dropout(dropout)

    def forward(self, q, image, need_attn=False):
        x = self.norm1(q)
        q = q + self.drop(self.self_attn(x, x, x, need_weights=False)[0])
        out, attn = self.cross_attn(self.norm2(q), image, image, need_weights=need_attn)
        q = q + self.drop(out)
        return q + self.drop(self.ff(self.norm3(q))), attn


class QueryBridge(nn.Module):
    """Learned queries read the patch tokens through cross-attention (a small Q-Former-style resampler)."""

    def __init__(self, vision_dim, out_dim, tokens=32, width=512, layers=2, heads=8, dropout=0.1, vision_tokens=256):
        super().__init__()
        self.in_norm = nn.LayerNorm(vision_dim)
        self.in_proj = nn.Linear(vision_dim, width)
        self.pos = nn.Parameter(torch.zeros(vision_tokens, width))
        self.queries = nn.Parameter(torch.randn(tokens, width) * 0.02)
        self.blocks = nn.ModuleList(Block(width, heads, dropout) for _ in range(layers))
        self.out_norm = nn.LayerNorm(width)
        self.last = nn.Linear(width, out_dim)

    def forward(self, feats, need_attn=False):
        image = self.in_proj(self.in_norm(feats)) + self.pos
        q = self.queries.unsqueeze(0).expand(feats.shape[0], -1, -1)
        attns = []
        for block in self.blocks:
            q, attn = block(q, image, need_attn)
            attns.append(attn)
        out = self.last(self.out_norm(q))
        return (out, attns) if need_attn else out


class PooledMLPBridge(nn.Module):
    """Average-pool the patch grid down to `tokens` cells, then a 2-layer MLP (LLaVA-1.5 style)."""

    def __init__(self, vision_dim, out_dim, tokens=16, vision_tokens=256):
        super().__init__()
        self.grid, self.side = round(vision_tokens ** 0.5), round(tokens ** 0.5)
        assert self.grid ** 2 == vision_tokens and self.side ** 2 == tokens, "token counts must be squares"
        self.in_norm = nn.LayerNorm(vision_dim)
        self.hidden = nn.Linear(vision_dim, out_dim)
        self.last = nn.Linear(out_dim, out_dim)

    def forward(self, feats, need_attn=False):
        b, _, d = feats.shape
        x = feats.transpose(1, 2).reshape(b, d, self.grid, self.grid)
        x = F.adaptive_avg_pool2d(x, self.side).flatten(2).transpose(1, 2)
        out = self.last(F.gelu(self.hidden(self.in_norm(x))))
        return (out, []) if need_attn else out


def build_bridge(cfg, out_dim):
    b, v = cfg["bridge"], cfg["vision"]
    if b["type"] == "mlp":
        return PooledMLPBridge(v["dim"], out_dim, b["tokens"], v["tokens"])
    return QueryBridge(v["dim"], out_dim, b["tokens"], b["width"], b["layers"], b["heads"], b["dropout"], v["tokens"])
