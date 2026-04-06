import math

import torch
import torch.nn.functional as F
from torch import nn


class ReliabilityHead(nn.Module):
    """Estimate per-slot reliability from focus entropy and slot consistency."""

    def __init__(self, dim: int, hidden_dim: int = 256, eps: float = 1e-6):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eps = eps

        self.hz_norm = nn.LayerNorm(dim)
        self.h_tilde_norm = nn.LayerNorm(dim)

        input_dim = dim * 2 + 2  # hz + h_tilde + focus + cosine similarity
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def compute_focus(self, slot_attention: torch.Tensor) -> torch.Tensor:
        if slot_attention.dim() != 3:
            raise ValueError(f"slot_attention must have shape [B, K, N], got {tuple(slot_attention.shape)}")
        # 归一化 
        slot_attention = slot_attention / slot_attention.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        # 计算熵 H(p) = -∑_i p_i * log(p_i)，如果大于0则计算熵，否则就取0
        entropy_terms = torch.where(
            slot_attention > 0,
            slot_attention * slot_attention.clamp_min(self.eps).log(),
            torch.zeros_like(slot_attention),
        )
        entropy = -entropy_terms.sum(dim=-1)

        num_tokens = slot_attention.size(-1)
        if num_tokens <= 1:
            return torch.ones_like(entropy)

        return 1.0 - entropy / math.log(num_tokens)

    def forward(self, hz: torch.Tensor, h_tilde: torch.Tensor, slot_attention: torch.Tensor):
        if hz.dim() != 3:
            raise ValueError(f"hz must have shape [B, K, C], got {tuple(hz.shape)}")
        if h_tilde.dim() != 3:
            raise ValueError(f"h_tilde must have shape [B, K, C], got {tuple(h_tilde.shape)}")
        if hz.shape != h_tilde.shape:
            raise ValueError("hz and h_tilde must have the same shape")
        if hz.size(-1) != self.dim:
            raise ValueError(f"Expected slot dim {self.dim}, got {hz.size(-1)}")

        batch_size, num_slots, _ = hz.shape
        if (
            slot_attention.dim() != 3
            or slot_attention.size(0) != batch_size
            or slot_attention.size(1) != num_slots
        ):
            raise ValueError("slot_attention must align with hz on batch and slot dimensions")

        slot_focus = self.compute_focus(slot_attention)
        slot_similarity = F.cosine_similarity(hz, h_tilde, dim=-1, eps=self.eps)
        hz_mlp = self.hz_norm(hz)
        h_tilde_mlp = self.h_tilde_norm(h_tilde)

        fused = torch.cat(
            [
                hz_mlp,
                h_tilde_mlp,
                slot_focus.unsqueeze(-1),
                slot_similarity.unsqueeze(-1),
            ],
            dim=-1,
        )
        reliability = torch.sigmoid(self.mlp(fused)).squeeze(-1)

        aux = {
            "slot_focus": slot_focus,
            "slot_similarity": slot_similarity,
        }
        return reliability, aux
