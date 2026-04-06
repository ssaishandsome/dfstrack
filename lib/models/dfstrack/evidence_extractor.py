from typing import Optional, Tuple, Union

import torch
from torch import nn


def _check_search_tokens(search_tokens: torch.Tensor):
    """Validate search token shape [B, N, C]."""
    if search_tokens.dim() != 3:
        raise ValueError(
            f"search_tokens must have shape [B, N, C], got {tuple(search_tokens.shape)}"
        )


def _safe_masked_mean(tokens: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute a masked mean over token dimension.

    Args:
        tokens: [B, N, C]
        mask: [B, N] with values in {0, 1} or bool

    Returns:
        pooled: [B, C]
    """
    mask = mask.float()
    denom = mask.sum(dim=1, keepdim=True).clamp_min(eps)  # [B, 1]
    weighted = tokens * mask.unsqueeze(-1)  # [B, N, C]
    return weighted.sum(dim=1) / denom


def _resolve_feat_hw(
    num_tokens: int,
    feat_size: Optional[Union[int, Tuple[int, int]]],
) -> Optional[Tuple[int, int]]:
    """Resolve token grid size from N and optional feature size."""
    if feat_size is None:
        side = int(num_tokens ** 0.5)
        if side * side == num_tokens:
            return side, side
        return None

    if isinstance(feat_size, int):
        return feat_size, feat_size

    if isinstance(feat_size, tuple) and len(feat_size) == 2:
        return feat_size

    raise ValueError("feat_size must be None, int, or a tuple (H, W)")


def _index_to_hw(index: torch.Tensor, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert flattened token index [B] to (row, col)."""
    row = index // width
    col = index % width
    return row, col


def _ring_mask_from_index(
    target_index: torch.Tensor,
    num_tokens: int,
    feat_hw: Optional[Tuple[int, int]],
    target_radius: int,
    context_radius: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build target and context masks from token center index.

    Returns:
        target_mask: [B, N]
        context_mask: [B, N]
    """
    batch_size = target_index.shape[0]

    if feat_hw is None:
        # Fallback without grid structure:
        # target = one-hot token, context = all other tokens.
        target_mask = torch.zeros(batch_size, num_tokens, device=target_index.device, dtype=torch.float32)
        target_mask.scatter_(1, target_index.unsqueeze(1), 1.0)
        context_mask = 1.0 - target_mask
        return target_mask, context_mask

    height, width = feat_hw
    rows = torch.arange(height, device=target_index.device).view(1, height, 1)
    cols = torch.arange(width, device=target_index.device).view(1, 1, width)

    center_r, center_c = _index_to_hw(target_index, width)
    center_r = center_r.view(batch_size, 1, 1)
    center_c = center_c.view(batch_size, 1, 1)

    dist_r = (rows - center_r).abs()
    dist_c = (cols - center_c).abs()
    chebyshev_dist = torch.maximum(dist_r, dist_c)

    target_mask_2d = (chebyshev_dist <= target_radius).float()
    context_outer_2d = (chebyshev_dist <= context_radius).float()
    context_mask_2d = (context_outer_2d - target_mask_2d).clamp_min(0.0)

    target_mask = target_mask_2d.view(batch_size, -1)
    context_mask = context_mask_2d.view(batch_size, -1)
    return target_mask, context_mask


def _box_to_masks(
    target_box: torch.Tensor,
    feat_hw: Tuple[int, int],
    context_radius: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build target/context masks from normalized xyxy boxes.

    Args:
        target_box: [B, 4], normalized xyxy in [0, 1].
        feat_hw: (H, W)
        context_radius: dilation radius on token grid.

    Returns:
        target_mask: [B, N]
        context_mask: [B, N]
    """
    if target_box.dim() != 2 or target_box.size(-1) != 4:
        raise ValueError(f"target_box must have shape [B, 4], got {tuple(target_box.shape)}")

    batch_size = target_box.size(0)
    height, width = feat_hw
    device = target_box.device

    x1, y1, x2, y2 = target_box.unbind(dim=-1)
    x1 = (x1.clamp(0, 1) * (width - 1)).floor().long()
    y1 = (y1.clamp(0, 1) * (height - 1)).floor().long()
    x2 = (x2.clamp(0, 1) * (width - 1)).ceil().long()
    y2 = (y2.clamp(0, 1) * (height - 1)).ceil().long()

    rows = torch.arange(height, device=device).view(1, height, 1)
    cols = torch.arange(width, device=device).view(1, 1, width)

    x1v = x1.view(batch_size, 1, 1)
    y1v = y1.view(batch_size, 1, 1)
    x2v = x2.view(batch_size, 1, 1)
    y2v = y2.view(batch_size, 1, 1)

    target_mask_2d = (
        (rows >= y1v) & (rows <= y2v) &
        (cols >= x1v) & (cols <= x2v)
    ).float()

    x1c = (x1 - context_radius).clamp(min=0)
    y1c = (y1 - context_radius).clamp(min=0)
    x2c = (x2 + context_radius).clamp(max=width - 1)
    y2c = (y2 + context_radius).clamp(max=height - 1)

    x1cv = x1c.view(batch_size, 1, 1)
    y1cv = y1c.view(batch_size, 1, 1)
    x2cv = x2c.view(batch_size, 1, 1)
    y2cv = y2c.view(batch_size, 1, 1)

    context_outer_2d = (
        (rows >= y1cv) & (rows <= y2cv) &
        (cols >= x1cv) & (cols <= x2cv)
    ).float()
    context_mask_2d = (context_outer_2d - target_mask_2d).clamp_min(0.0)

    return target_mask_2d.view(batch_size, -1), context_mask_2d.view(batch_size, -1)


class EvidenceExtractor(nn.Module):
    """Extract target, context, and motion evidence from search tokens.

    Version 1 behavior:
    - vt comes from a target-centered pooled feature, or a target token if provided
    - ct comes from nearby surrounding tokens excluding the target center
    - mt = vt - prev_vt
    - if prev_vt is None, mt is zeros

    Inputs:
        search_tokens: [B, N, C]
        prev_vt: [B, C] or None
        target_index: [B] flattened token index, optional
        target_box: [B, 4] normalized xyxy box, optional
        feat_size: int or (H, W), optional

    Returns:
        vt: [B, C]
        ct: [B, C]
        mt: [B, C]
    """

    def __init__(
        self,
        target_radius: int = 0,
        context_radius: int = 1,
        eps: float = 1e-6,
    ):
        super().__init__()
        if target_radius < 0:
            raise ValueError("target_radius must be >= 0")
        if context_radius < target_radius:
            raise ValueError("context_radius must be >= target_radius")

        self.target_radius = target_radius
        self.context_radius = context_radius
        self.eps = eps

    def forward(
        self,
        search_tokens: torch.Tensor,
        prev_vt: Optional[torch.Tensor] = None, # 上一个时间步的目标特征，如果没有则为 None
        target_index: Optional[torch.Tensor] = None,
        target_box: Optional[torch.Tensor] = None,
        feat_size: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _check_search_tokens(search_tokens)

        if (target_index is None) and (target_box is None):
            raise ValueError("Either target_index or target_box must be provided")

        batch_size, num_tokens, dim = search_tokens.shape
        feat_hw = _resolve_feat_hw(num_tokens, feat_size)

        if target_index is not None:
            if target_index.dim() == 2 and target_index.size(-1) == 1:
                target_index = target_index.squeeze(-1)
            if target_index.dim() != 1 or target_index.size(0) != batch_size:
                raise ValueError(f"target_index must have shape [B], got {tuple(target_index.shape)}")
            if (target_index < 0).any() or (target_index >= num_tokens).any():
                raise ValueError("target_index contains out-of-range indices")

            target_mask, context_mask = _ring_mask_from_index(
                target_index=target_index,
                num_tokens=num_tokens,
                feat_hw=feat_hw,
                target_radius=self.target_radius,
                context_radius=self.context_radius,
            )
        else:
            if feat_hw is None:
                raise ValueError("target_box requires resolvable feature grid size")
            target_mask, context_mask = _box_to_masks(
                target_box=target_box,
                feat_hw=feat_hw,
                context_radius=self.context_radius,
            )

        # vt: target-centered feature
        vt = _safe_masked_mean(search_tokens, target_mask, eps=self.eps)  # [B, C]

        # ct: nearby context excluding target center; fallback to global context if empty
        context_mass = context_mask.sum(dim=1, keepdim=True)  # [B, 1]
        global_context_mask = 1.0 - target_mask
        ct_local = _safe_masked_mean(search_tokens, context_mask, eps=self.eps)
        ct_global = _safe_masked_mean(search_tokens, global_context_mask, eps=self.eps)
        use_local = (context_mass > 0).float()
        ct = use_local * ct_local + (1.0 - use_local) * ct_global

        # mt: simple motion cue from temporal difference
        if prev_vt is None:
            mt = torch.zeros_like(vt)
        else:
            if prev_vt.shape != (batch_size, dim):
                raise ValueError(f"prev_vt must have shape {(batch_size, dim)}, got {tuple(prev_vt.shape)}")
            mt = vt - prev_vt

        return vt, ct, mt


if __name__ == "__main__":
    """
    vt
        如果给的是 target_index
        默认取 target center token
        如果 target_radius > 0，就取中心附近一小块 token 做 pooling
        如果给的是 target_box
        就对 box 覆盖到的 token 区域做平均
    ct
        取目标周围的“ring”区域，也就是附近 context
        会显式排除 target 中心区域
        如果局部 context 恰好为空，会退回到“除目标外的全局上下文”
    mt
        mt = vt - prev_vt
        如果 prev_vt is None，直接返回全零
    """
    batch_size, num_tokens, dim = 2, 256, 8
    search_tokens = torch.randn(batch_size, num_tokens, dim)
    target_index = torch.tensor([42, 99])
    prev_vt = torch.randn(batch_size, dim)

    extractor = EvidenceExtractor(target_radius=0, context_radius=2)
    vt, ct, mt = extractor(
        search_tokens=search_tokens,
        prev_vt=prev_vt,
        target_index=target_index,
        feat_size=16,
    )

    print("vt shape:", tuple(vt.shape))
    print("ct shape:", tuple(ct.shape))
    print("mt shape:", tuple(mt.shape))
