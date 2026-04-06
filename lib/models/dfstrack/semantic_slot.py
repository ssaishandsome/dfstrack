import torch
from torch import nn

from lib.models.dfstrack.reliability_head import ReliabilityHead


class CrossAttention(nn.Module):
    """A lightweight cross-attention layer that can optionally expose attention maps."""

    def __init__(self, dim: int, num_heads: int = 8, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)

        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_drop = nn.Dropout(proj_drop)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, length, _ = x.shape
        return x.view(batch_size, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor = None,
        return_attn: bool = False,
    ):
        if query.dim() != 3:
            raise ValueError(f"query must have shape [B, Lq, C], got {tuple(query.shape)}")
        if context.dim() != 3:
            raise ValueError(f"context must have shape [B, Lk, C], got {tuple(context.shape)}")
        if query.size(0) != context.size(0):
            raise ValueError("query and context must have the same batch size")
        if query.size(-1) != self.dim or context.size(-1) != self.dim:
            raise ValueError(f"Expected feature dim {self.dim}")

        if context_mask is not None:
            if context_mask.shape != context.shape[:2]:
                raise ValueError(
                    f"context_mask must have shape {tuple(context.shape[:2])}, got {tuple(context_mask.shape)}"
                )
            context_mask = context_mask.bool()
            fully_masked = context_mask.all(dim=1)
            if fully_masked.any():
                context_mask = context_mask.clone()
                context_mask[fully_masked, 0] = False

        q = self._reshape_heads(self.query_proj(self.query_norm(query)))
        k = self._reshape_heads(self.key_proj(self.context_norm(context)))
        v = self._reshape_heads(self.value_proj(self.context_norm(context)))

        attn_logits = torch.matmul(q * self.scale, k.transpose(-2, -1))
        if context_mask is not None:
            attn_logits = attn_logits.masked_fill(context_mask[:, None, None, :], float("-inf"))

        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(query.size(0), query.size(1), self.dim)
        out = self.out_drop(self.out_proj(out))

        if return_attn:
            return out, attn.mean(dim=1)
        return out


class ResidualCrossAttentionBlock(nn.Module):
    """Cross-attention block with residual MLP refinement."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        hidden_dim: int = 256,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.cross_attn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(proj_drop),
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor = None,
        return_attn: bool = False,
    ):
        if return_attn:
            delta, attn = self.cross_attn(
                query=query,
                context=context,
                context_mask=context_mask,
                return_attn=True,
            )
        else:
            delta = self.cross_attn(
                query=query,
                context=context,
                context_mask=context_mask,
                return_attn=False,
            )
            attn = None

        x = query + delta
        x = x + self.mlp(self.mlp_norm(x))

        if return_attn:
            return x, attn
        return x


class LanguageSlotInitializer(nn.Module):
    """Initialize semantic slots from text token features."""

    def __init__(
        self,
        num_slots: int,
        dim: int,
        num_heads: int = 8,
        hidden_dim: int = 256,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")

        self.num_slots = num_slots
        self.dim = dim
        self.slot_queries = nn.Parameter(torch.randn(1, num_slots, dim) * (dim ** -0.5))
        self.cross_block = ResidualCrossAttentionBlock(
            dim=dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def forward(self, text_tokens: torch.Tensor, text_mask: torch.Tensor = None):
        if text_tokens.dim() != 3:
            raise ValueError(f"text_tokens must have shape [B, Lt, C], got {tuple(text_tokens.shape)}")
        if text_tokens.size(-1) != self.dim:
            raise ValueError(f"Expected text dim {self.dim}, got {text_tokens.size(-1)}")

        slot_queries = self.slot_queries.expand(text_tokens.size(0), -1, -1)
        slots, attn = self.cross_block(
            query=slot_queries,
            context=text_tokens,
            context_mask=text_mask,
            return_attn=True,
        )
        return slots, {"text_slot_attention": attn}


class SlotSearchInteraction(nn.Module):
    """Update template-guided slots with template/search evidence."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        hidden_dim: int = 256,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.template_refiner = ResidualCrossAttentionBlock(
            dim=dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.search_attn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Sequential(
            nn.Linear(dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(proj_drop),
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(proj_drop),
        )

    def forward(self, template_slots: torch.Tensor, template_tokens: torch.Tensor, search_tokens: torch.Tensor):
        template_enhanced, template_attn = self.template_refiner(
            query=template_slots,
            context=template_tokens,
            return_attn=True,
        )
        search_ctx, slot_attention = self.search_attn(
            query=template_enhanced,
            context=search_tokens,
            return_attn=True,
        )

        fused = self.out_proj(torch.cat([template_slots, template_enhanced, search_ctx], dim=-1))
        updated_slots = self.out_norm(template_slots + fused)
        updated_slots = updated_slots + self.ffn(self.ffn_norm(updated_slots))

        aux = {
            "template_slot_attention": template_attn,
            "slot_attention": slot_attention,
            "template_enhanced_slots": template_enhanced,
            "slot_search_context": search_ctx,
        }
        return updated_slots, aux


class SearchFeatureModulator(nn.Module):
    """Modulate search tokens with corrected semantic slots."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        hidden_dim: int = 256,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.cross_block = ResidualCrossAttentionBlock(
            dim=dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def forward(self, search_tokens: torch.Tensor, slots: torch.Tensor):
        modulated_search, attn = self.cross_block(
            query=search_tokens,
            context=slots,
            return_attn=True,
        )
        return modulated_search, {"search_slot_attention": attn}


class SemanticSlotTracker(nn.Module):
    """Semantic-slot pipeline helpers for text init, template constraint, correction, and search modulation."""

    def __init__(
        self,
        num_slots: int,
        dim: int,
        num_heads: int = 8,
        hidden_dim: int = 256,
        reliability_hidden_dim: int = 256,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.slot_initializer = LanguageSlotInitializer(
            num_slots=num_slots,
            dim=dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.template_guidance = ResidualCrossAttentionBlock(
            dim=dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.reliability_head = ReliabilityHead(
            dim=dim,
            hidden_dim=reliability_hidden_dim,
        )
        self.search_modulator = SearchFeatureModulator(
            dim=dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    @staticmethod
    def slot_attention_to_assignment(slot_attention: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if slot_attention.dim() != 3:
            raise ValueError(f"slot_attention must have shape [B, K, N], got {tuple(slot_attention.shape)}")
        assignment = slot_attention.transpose(1, 2)
        return assignment / assignment.sum(dim=-1, keepdim=True).clamp_min(eps)

    def initialize_slots(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor = None,
    ):
        return self.slot_initializer(text_tokens=text_tokens, text_mask=text_mask)

    def constrain_slots(
        self,
        slot_prior: torch.Tensor,
        template_tokens: torch.Tensor,
    ):
        template_slots, template_aux = self.template_guidance(
            query=slot_prior,
            context=template_tokens,
            return_attn=True,
        )
        return template_slots, {"template_slot_attention": template_aux}

    def correct_slots(
        self,
        template_slots: torch.Tensor,
        slot_candidate: torch.Tensor,
        slot_attention: torch.Tensor,
    ):
        reliability, reliability_aux = self.reliability_head(
            hz=template_slots,
            h_tilde=slot_candidate,
            slot_attention=slot_attention,
        )
        corrected_slots = (
            reliability.unsqueeze(-1) * slot_candidate
            + (1.0 - reliability).unsqueeze(-1) * template_slots
        )
        slot_assignment = self.slot_attention_to_assignment(slot_attention)

        aux = {
            "slot_candidate": slot_candidate,
            "slot_state": corrected_slots, # 根据可信度加权融合后的slot特征
            "slot_attention": slot_attention,
            "slot_assignment": slot_assignment, # 
            "slot_reliability": reliability,
            "slot_focus": reliability_aux["slot_focus"],
            "slot_similarity": reliability_aux["slot_similarity"],
        }
        return corrected_slots, aux

    def modulate_search(
        self,
        search_tokens: torch.Tensor,
        corrected_slots: torch.Tensor,
    ):
        modulated_search, modulation_aux = self.search_modulator(
            search_tokens=search_tokens,
            slots=corrected_slots,
        )
        return modulated_search, modulation_aux
