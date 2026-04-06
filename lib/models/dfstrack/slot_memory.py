from typing import Optional

import torch
from torch import nn


def _check_3d(x: torch.Tensor, name: str):
    """Validate a tensor has shape [B, *, C]."""
    if x.dim() != 3:
        raise ValueError(f"{name} must be 3D, got shape {tuple(x.shape)}")


class SlotVisualRetriever(nn.Module):
    """Retrieve slot-conditioned visual evidence from search tokens.

    Args:
        dim: Feature dimension C.

    Inputs:
        h_prev: Previous slot states with shape [B, K, C].
        search_tokens: Current search tokens with shape [B, M, C].

    Returns:
        u: Retrieved visual evidence with shape [B, K, C].
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")

        self.dim = dim
        self.slot_proj = nn.Linear(dim, dim)
        self.token_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

    def forward(self, h_prev: torch.Tensor, search_tokens: torch.Tensor) -> torch.Tensor:
        _check_3d(h_prev, "h_prev")
        _check_3d(search_tokens, "search_tokens")

        batch_size, num_slots, dim = h_prev.shape
        batch_size_tokens, num_tokens, dim_tokens = search_tokens.shape
        if batch_size_tokens != batch_size:
            raise ValueError("h_prev and search_tokens must have the same batch size")
        if dim != self.dim or dim_tokens != self.dim:
            raise ValueError(f"Expected feature dim {self.dim}, got {dim} and {dim_tokens}")

        # Queries from previous slot states: [B, K, C]
        q = self.slot_proj(h_prev)

        # Keys / values from current search tokens: [B, M, C]
        k = self.token_proj(search_tokens)
        v = self.value_proj(search_tokens)

        # Slot-token attention scores: [B, K, M]
        scores = torch.matmul(q, k.transpose(1, 2)) / (self.dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)

        # Retrieved visual evidence: [B, K, C]
        u = torch.matmul(attn, v)
        return u


class TemporalSlotMemory(nn.Module):
    """Update slot memory with stable and adaptive slot groups.

    Version 1 update rule:
        delta = update_mlp([h_prev, u])
        candidate = h_prev + delta
        gated_candidate = reliability * candidate + (1 - reliability) * h_prev
        h_new = alpha * h_prev + (1 - alpha) * gated_candidate

    Args:
        dim: Feature dimension C.
        num_slots: Number of slots K.
        num_stable_slots: Number of stable-biased slots from the front of K.
        stable_alpha: Retention rate for stable-biased slots.
            Larger alpha means stronger memory retention and slower update.
        adaptive_alpha: Retention rate for adaptive slots.
            Smaller alpha means weaker memory retention and faster update.
        use_fixed_alpha: If True, use fixed alpha values from config.
            If False, learn per-slot alpha values.

    Inputs:
        h_prev: Previous slot states [B, K, C].
        u: Retrieved slot-conditioned evidence [B, K, C].
        reliability: Reliability weights [B, K] in [0, 1].

    Returns:
        h_new: Updated slot states [B, K, C].
        candidate: Candidate slot updates [B, K, C].
        alpha: Per-slot update rates [K].
    """

    def __init__(
        self,
        dim: int,
        num_slots: int,
        num_stable_slots: int = 3,
        stable_alpha: float = 0.8,
        adaptive_alpha: float = 0.4,
        use_fixed_alpha: bool = True,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")
        if not (0 <= num_stable_slots <= num_slots):
            raise ValueError("num_stable_slots must be in [0, num_slots]")
        if not (0.0 <= stable_alpha <= 1.0):
            raise ValueError("stable_alpha must be in [0, 1]")
        if not (0.0 <= adaptive_alpha <= 1.0):
            raise ValueError("adaptive_alpha must be in [0, 1]")

        self.dim = dim
        self.num_slots = num_slots
        self.num_stable_slots = num_stable_slots
        self.num_adaptive_slots = num_slots - num_stable_slots
        self.stable_alpha = stable_alpha
        self.adaptive_alpha = adaptive_alpha
        self.use_fixed_alpha = use_fixed_alpha

        # Candidate delta network: [B, K, 2C] -> [B, K, C]
        self.update_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

        if not self.use_fixed_alpha:
            # Learn one update rate per slot. Sigmoid constrains to [0, 1].
            init_alpha = torch.empty(num_slots)
            if num_stable_slots > 0:
                init_alpha[:num_stable_slots] = stable_alpha
            if self.num_adaptive_slots > 0:
                init_alpha[num_stable_slots:] = adaptive_alpha
            init_alpha = init_alpha.clamp(1e-4, 1 - 1e-4)
            self.alpha_logits = nn.Parameter(torch.logit(init_alpha))
        else:
            self.register_parameter("alpha_logits", None)

    def get_alpha(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return per-slot update rates with shape [K]."""
        if self.use_fixed_alpha:
            alpha = torch.empty(self.num_slots, device=device, dtype=dtype)
            if self.num_stable_slots > 0:
                alpha[:self.num_stable_slots] = self.stable_alpha
            if self.num_adaptive_slots > 0:
                alpha[self.num_stable_slots:] = self.adaptive_alpha
            return alpha

        return torch.sigmoid(self.alpha_logits).to(device=device, dtype=dtype)

    def forward(
        self,
        h_prev: torch.Tensor,
        u: torch.Tensor,
        reliability: torch.Tensor,
    ):
        _check_3d(h_prev, "h_prev")
        _check_3d(u, "u")

        if reliability.dim() != 2:
            raise ValueError(f"reliability must have shape [B, K], got {tuple(reliability.shape)}")

        batch_size, num_slots, dim = h_prev.shape
        if u.shape != (batch_size, num_slots, dim):
            raise ValueError(f"u must have shape {(batch_size, num_slots, dim)}, got {tuple(u.shape)}")
        if reliability.shape != (batch_size, num_slots):
            raise ValueError(
                f"reliability must have shape {(batch_size, num_slots)}, got {tuple(reliability.shape)}"
            )
        if num_slots != self.num_slots or dim != self.dim:
            raise ValueError(
                f"Expected h_prev/u to have shape [B, {self.num_slots}, {self.dim}], got {tuple(h_prev.shape)}"
            )

        # Predict a residual update from previous memory and current evidence.
        delta = self.update_mlp(torch.cat([h_prev, u], dim=-1))
        candidate = h_prev + delta

        # Reliability gate controls how much current evidence is trusted: [B, K, 1]
        reliability = reliability.clamp(0.0, 1.0).unsqueeze(-1)
        gated_candidate = reliability * candidate + (1.0 - reliability) * h_prev

        # Per-slot update rates: [K] -> [1, K, 1]
        alpha = self.get_alpha(device=h_prev.device, dtype=h_prev.dtype)
        alpha_broadcast = alpha.view(1, num_slots, 1)

        # alpha large -> stronger memory retention / slower update.
        h_new = alpha_broadcast * h_prev + (1.0 - alpha_broadcast) * gated_candidate

        return h_new, candidate, alpha
