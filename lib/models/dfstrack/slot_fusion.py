import torch
from torch import nn


class SlotFusion(nn.Module):
    """Version-1 fusion from slot memory to target representation.

    Inputs:
        vt: Current target feature with shape [B, C].
        h: Slot states with shape [B, K, C].
        r: Slot reliability weights with shape [B, K].

    Returns:
        fused target representation with shape [B, C].
    """

    def __init__(self, dim: int, hidden_dim: int = 256):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.dim = dim
        self.hidden_dim = hidden_dim

        self.fuse_mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, vt: torch.Tensor, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if vt.dim() != 2:
            raise ValueError(f"vt must have shape [B, C], got {tuple(vt.shape)}")
        if h.dim() != 3:
            raise ValueError(f"h must have shape [B, K, C], got {tuple(h.shape)}")
        if r.dim() != 2:
            raise ValueError(f"r must have shape [B, K], got {tuple(r.shape)}")

        batch_size, dim = vt.shape
        batch_size_h, num_slots, dim_h = h.shape
        batch_size_r, num_slots_r = r.shape

        if batch_size_h != batch_size or batch_size_r != batch_size:
            raise ValueError("vt, h, and r must have the same batch size")
        if dim != self.dim or dim_h != self.dim:
            raise ValueError(f"Expected feature dim {self.dim}, got {dim} and {dim_h}")
        if num_slots_r != num_slots:
            raise ValueError("r and h must agree on the slot dimension")

        r = r.clamp(0.0, 1.0).unsqueeze(-1)             # [B, K, 1]
        z = (r * h).sum(dim=1)                          # [B, C]

        delta = self.fuse_mlp(torch.cat([vt, z], dim=-1))
        fused = vt + delta
        return self.norm(fused)
