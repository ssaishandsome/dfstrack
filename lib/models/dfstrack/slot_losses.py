import torch
import torch.nn.functional as F


def _check_3d_tensor(x: torch.Tensor, name: str):
    """Validate that a tensor is 3D."""
    if x.dim() != 3:
        raise ValueError(f"{name} must have shape [B, N, K] or [B, K, C], got {tuple(x.shape)}")


def _safe_normalize(x: torch.Tensor, dim: int, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize a tensor along a given dimension with numerical stability."""
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def _mean_off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    """Return the mean of off-diagonal entries for a batch of square matrices.

    Expected shape:
        matrix: [B, K, K]
    """
    batch_size, num_slots, _ = matrix.shape
    eye = torch.eye(num_slots, device=matrix.device, dtype=matrix.dtype).unsqueeze(0)
    off_diag = matrix * (1.0 - eye)
    denom = max(num_slots * (num_slots - 1), 1)
    return off_diag.sum() / (batch_size * denom)


def slot_attention_diversity_loss(assign: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Penalize similar slot assignment patterns.

    Args:
        assign: Soft token-to-slot assignment with shape [B, N, K].
            It is expected that assign sums to 1 over the last dimension.
        eps: Small constant for stability.

    Returns:
        A scalar tensor. Lower is better.
    """
    _check_3d_tensor(assign, "assign")

    # Compare slot attention maps across tokens inside each sample.
    # [B, N, K] -> [B, K, N]
    slot_maps = assign.transpose(1, 2)
    slot_maps = _safe_normalize(slot_maps, dim=-1, eps=eps)

    # Slot-slot cosine similarity: [B, K, K]
    similarity = torch.matmul(slot_maps, slot_maps.transpose(1, 2))
    return _mean_off_diagonal(similarity)


def slot_orthogonality_loss(slots: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Encourage slot embeddings to be mutually orthogonal.

    Args:
        slots: Slot features with shape [B, K, C].
        eps: Small constant for stability.

    Returns:
        A scalar tensor. Lower is better.
    """
    _check_3d_tensor(slots, "slots")

    slots = _safe_normalize(slots, dim=-1, eps=eps)
    gram = torch.matmul(slots, slots.transpose(1, 2))  # [B, K, K]

    num_slots = gram.size(1)
    eye = torch.eye(num_slots, device=gram.device, dtype=gram.dtype).unsqueeze(0)
    return F.mse_loss(gram, eye.expand_as(gram))


def slot_balance_loss(assign: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Encourage balanced slot usage across the batch.

    Args:
        assign: Soft token-to-slot assignment with shape [B, N, K].
        eps: Small constant for stability.

    Returns:
        A scalar tensor. Lower is better.
    """
    _check_3d_tensor(assign, "assign")

    # Batch-level slot mass: [K]
    slot_mass = assign.sum(dim=(0, 1))
    slot_prob = slot_mass / slot_mass.sum().clamp_min(eps)

    num_slots = slot_prob.numel()
    target = torch.full_like(slot_prob, 1.0 / num_slots)
    return F.mse_loss(slot_prob, target)


if __name__ == "__main__":
    batch_size, num_tokens, num_slots, dim = 2, 16, 4, 8

    logits = torch.randn(batch_size, num_tokens, num_slots)
    assign = torch.softmax(logits, dim=-1)
    slots = torch.randn(batch_size, num_slots, dim)

    loss_div = slot_attention_diversity_loss(assign)
    loss_ortho = slot_orthogonality_loss(slots)
    loss_balance = slot_balance_loss(assign)

    print("assign shape:", tuple(assign.shape))
    print("slots shape:", tuple(slots.shape))
    print("diversity loss:", float(loss_div))
    print("orthogonality loss:", float(loss_ortho))
    print("balance loss:", float(loss_balance))
