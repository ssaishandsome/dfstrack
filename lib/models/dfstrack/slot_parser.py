import torch
from torch import nn


class SlotParser(nn.Module):
    """Parse token features into a fixed number of learnable slots.

    Args:
        num_slots: Number of slots K.
        dim: Feature dimension C.
        eps: Small value for stable normalization.
        temperature: Softmax temperature for token-to-slot assignment.

    Inputs:
        tokens: Tensor with shape [B, N, C].

    Returns:
        slots: Tensor with shape [B, K, C].
        assignment: Tensor with shape [B, N, K], normalized over K.
    """

    def __init__(self, num_slots: int, dim: int, eps: float = 1e-6, temperature: float = 1.0):
        super().__init__()
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")
        if dim <= 0:
            raise ValueError("dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.num_slots = num_slots
        self.dim = dim
        self.eps = eps
        self.temperature = temperature

        self.slot_queries = nn.Parameter(torch.randn(num_slots, dim) * (dim ** -0.5))
        self.token_proj = nn.Linear(dim, dim)
        self.slot_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

    def forward(self, tokens: torch.Tensor):
        if tokens.dim() != 3:
            raise ValueError(f"tokens must have shape [B, N, C], got {tuple(tokens.shape)}")

        _, _, dim = tokens.shape
        if dim != self.dim:
            raise ValueError(f"Expected token dim {self.dim}, got {dim}")

        token_feats = self.token_proj(tokens)  # [B, N, C]
        slot_feats = self.slot_proj(self.slot_queries)  # [K, C]

        scores = torch.einsum("bnc,kc->bnk", token_feats, slot_feats)
        scores = scores / ((self.dim ** 0.5) * self.temperature)

        # Normalize over slots for each token.
        assignment = torch.softmax(scores, dim=-1)  # [B, N, K]

        # Project values separately before aggregation.
        value_feats = self.value_proj(tokens)  # [B, N, C]

        # Aggregate tokens into slots, then normalize by slot mass.
        weighted_sum = torch.einsum("bnk,bnc->bkc", assignment, value_feats)  # [B, K, C]
        slot_mass = assignment.sum(dim=1).unsqueeze(-1)  # [B, K, 1] 分配到slot的总概率
        slots = weighted_sum / slot_mass.clamp_min(self.eps)  # 归一化

        return slots, assignment
