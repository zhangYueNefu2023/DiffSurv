import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


TensorDict = Dict[str, torch.Tensor]


class MultiOmicsTokenizer(nn.Module):
    """Map each omics modality into a shared latent-token space."""

    def __init__(self, mrna_dim: int, mirna_dim: int, protein_dim: int, embed_dim: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.projections = nn.ModuleDict(
            {
                "mrna": nn.Linear(mrna_dim, embed_dim),
                "mirna": nn.Linear(mirna_dim, embed_dim),
                "protein": nn.Linear(protein_dim, embed_dim),
            }
        )
        self.modality_embedding = nn.Parameter(torch.randn(1, 3, embed_dim) * 0.02)

    def forward(self, x: TensorDict) -> torch.Tensor:
        tokens = torch.stack(
            [
                self.projections["mrna"](x["mrna"]),
                self.projections["mirna"](x["mirna"]),
                self.projections["protein"](x["protein"]),
            ],
            dim=1,
        )
        return tokens + self.modality_embedding


class AdaLN(nn.Module):
    """Adaptive layer normalization conditioned on cancer type and diffusion time."""

    def __init__(self, embed_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.affine = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, embed_dim * 2))

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.affine(condition).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Transformer block with adaptive layer normalization."""

    def __init__(self, embed_dim: int, num_heads: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.adaln_attn = AdaLN(embed_dim, cond_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.adaln_mlp = AdaLN(embed_dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        h = self.adaln_attn(x, condition)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        h = self.adaln_mlp(x, condition)
        return x + self.mlp(h)


class DiffSurvModel(nn.Module):
    """Conditional latent diffusion model for multi-omics survival prediction."""

    def __init__(
        self,
        mrna_dim: int,
        mirna_dim: int,
        protein_dim: int,
        num_cancers: int,
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.tokenizer = MultiOmicsTokenizer(mrna_dim, mirna_dim, protein_dim, embed_dim)
        self.cancer_embedding = nn.Embedding(num_cancers, embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(embed_dim, num_heads, embed_dim, dropout=dropout) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.noise_head = nn.Linear(embed_dim, embed_dim)
        self.survival_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.embed_dim // 2
        scale = math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -scale)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if emb.shape[-1] < self.embed_dim:
            emb = torch.nn.functional.pad(emb, (0, self.embed_dim - emb.shape[-1]))
        return self.time_mlp(emb)

    def condition(self, cancer_id: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.cancer_embedding(cancer_id.long()) + self.time_embedding(t)

    def forward_latent(
        self, z_t: torch.Tensor, cancer_id: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = z_t
        cond = self.condition(cancer_id, t)
        for block in self.blocks:
            h = block(h, cond)
        h = self.final_norm(h)
        pred_noise = self.noise_head(h)
        risk_score = self.survival_head(h.mean(dim=1))
        return pred_noise, risk_score

    def forward(
        self, x: TensorDict, cancer_id: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.tokenizer(x)
        if t is None:
            t = torch.zeros(z.shape[0], device=z.device, dtype=torch.float32)
        return self.forward_latent(z, cancer_id, t)
