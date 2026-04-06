import os

import torch
from torch import nn

from transformers import RobertaModel, RobertaTokenizerFast

from lib.models.backbones.fast_itpn import (
    fast_itpn_base_3324_patch16_224,
    fast_itpn_large_2240_patch16_256,
)
from lib.models.dfstrack.semantic_slot import SemanticSlotTracker
from lib.models.layers.head import build_box_head
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.misc import NestedTensor


class DFSTrack(nn.Module):
    """A clean visual-language tracking baseline with shared visual/text encoders."""

    def __init__(self, backbone, box_head, tokenizer, text_encoder, head_type="CENTER", dim=512, cfg=None):
        super().__init__()
        self.backbone = backbone
        self.box_head = box_head
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.head_type = head_type
        self.dim = dim
        self.cfg = cfg

        if head_type in ["CORNER", "CENTER"]:
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        self.cls_prompts_pos = nn.Embedding(num_embeddings=1, embedding_dim=self.dim)
        self.text_adj = nn.Sequential(
            nn.Linear(768, self.dim, bias=True),
            nn.LayerNorm(self.dim, eps=1e-12),
            nn.Dropout(0.1),
        )

        dfs_cfg = getattr(cfg.MODEL, "DFS", None) if cfg is not None else None
        self.dfs_enabled = bool(getattr(dfs_cfg, "ENABLED", True))
        if self.dfs_enabled:
            self.slot_pos_embed = nn.Parameter(torch.zeros(1, dfs_cfg.NUM_SLOTS, self.dim))
            nn.init.trunc_normal_(self.slot_pos_embed, std=0.02)
            self.semantic_slots = SemanticSlotTracker(
                num_slots=dfs_cfg.NUM_SLOTS,
                dim=self.dim,
                num_heads=getattr(dfs_cfg, "NUM_HEADS", 8),
                hidden_dim=dfs_cfg.FUSION_HIDDEN_DIM,
                reliability_hidden_dim=dfs_cfg.RELIABILITY_HIDDEN_DIM,
                attn_drop=getattr(dfs_cfg, "ATTN_DROPOUT", 0.0),
                proj_drop=getattr(dfs_cfg, "PROJ_DROPOUT", 0.0),
            )

    def _split_backbone_tokens(self, tokens: torch.Tensor, template_token_lengths):
        if tokens.dim() != 3:
            raise ValueError(f"tokens must have shape [B, L, C], got {tuple(tokens.shape)}")

        template_tokens = []
        start = 0
        for length in template_token_lengths:
            end = start + length
            template_tokens.append(tokens[:, start:end])
            start = end
        search_tokens = tokens[:, start:]
        return template_tokens, search_tokens

    def forward_backbone_enc12(self, template, search, soft_token_template_mask):
        if not isinstance(template, (list, tuple)) or len(template) == 0:
            raise ValueError("template must be a non-empty list or tuple of template images")
        if len(template) != len(soft_token_template_mask):
            raise ValueError("template and soft_token_template_mask must have the same length")

        template_images = list(template)
        template_masks = list(soft_token_template_mask)
        tokens, _ = self.backbone.forward_features_pe(
            z=template_images,
            x=search,
            soft_token_template_mask=template_masks,
        )
        template_tokens, search_tokens = self._split_backbone_tokens(
            tokens=tokens,
            template_token_lengths=[mask.shape[1] for mask in template_masks],
        )
        return template_tokens, search_tokens

    def _build_stage3_pos_embed(self, batch_size: int, num_slot_tokens: int = 0):
        cls_prompts_pos = self.cls_prompts_pos.weight.unsqueeze(0)
        pos_chunks = [cls_prompts_pos]

        if num_slot_tokens > 0:
            if not self.dfs_enabled:
                raise ValueError("Slot positional embeddings require DFS to be enabled")
            if num_slot_tokens != self.slot_pos_embed.shape[1]:
                raise ValueError("num_slot_tokens must match the configured number of semantic slots")
            pos_chunks.append(self.slot_pos_embed)

        pos_chunks.extend([self.backbone.pos_embed_z, self.backbone.pos_embed_x])
        return torch.cat(pos_chunks, dim=1).repeat(batch_size, 1, 1)

    def forward_backbone_stage3(self, token_groups, x_pos, return_last_attn=False):
        tokens = torch.cat(token_groups, dim=1)
        return self.backbone.forward_features_stage3(
            tokens,
            None,
            x_pos,
            return_last_attn=return_last_attn,
        )

    def _split_stage3_outputs(
        self,
        tokens: torch.Tensor,
        num_slot_tokens: int,
        num_template_tokens: int,
    ):
        if tokens.dim() != 3:
            raise ValueError(f"tokens must have shape [B, L, C], got {tuple(tokens.shape)}")

        cursor = 1  # skip cls token
        slot_tokens = tokens[:, cursor:cursor + num_slot_tokens]
        cursor += num_slot_tokens
        template_tokens = tokens[:, cursor:cursor + num_template_tokens]
        cursor += num_template_tokens
        search_tokens = tokens[:, cursor:]
        return slot_tokens, template_tokens, search_tokens

    def _extract_slot_search_attention(
        self,
        attn: torch.Tensor,
        num_slot_tokens: int,
        num_template_tokens: int,
        num_search_tokens: int,
    ):
        if attn is None:
            raise ValueError("Stage-3 attention is required to compute slot reliability")
        if attn.dim() != 4:
            raise ValueError(f"attn must have shape [B, H, N, N], got {tuple(attn.shape)}")

        attn_mean = attn.mean(dim=1)
        slot_start = 1
        slot_end = slot_start + num_slot_tokens
        search_start = slot_end + num_template_tokens
        search_end = search_start + num_search_tokens
        return attn_mean[:, slot_start:slot_end, search_start:search_end]

    def _run_dfs_branch(
        self,
        template_tokens: torch.Tensor,
        search_tokens: torch.Tensor,
        text_features: NestedTensor,
        dfs_state: dict,
    ):
        slot_prior, init_aux = self.semantic_slots.initialize_slots(
            text_tokens=text_features.tensors,
            text_mask=text_features.mask,
        ) # 生成模板约束后的语义槽
        template_slots, template_aux = self.semantic_slots.constrain_slots(
            slot_prior=slot_prior,
            template_tokens=template_tokens,
        )
        # 为语义槽也构建位置编码，目前顺序为：cls token - slot tokens - template tokens - search tokens
        stage3_pos = self._build_stage3_pos_embed(
            batch_size=search_tokens.shape[0],
            num_slot_tokens=template_slots.shape[1],
        )
        stage3_tokens, stage3_aux = self.forward_backbone_stage3(
            token_groups=[template_slots, template_tokens, search_tokens],
            x_pos=stage3_pos,
            return_last_attn=True,
        )
        slot_candidate, stage3_template_tokens, stage3_search_tokens = self._split_stage3_outputs(
            stage3_tokens,
            num_slot_tokens=template_slots.shape[1],
            num_template_tokens=template_tokens.shape[1],
        ) # 选择出slot对应搜索图像特征对应的注意力分数
        slot_attention = self._extract_slot_search_attention(
            attn=stage3_aux["attn"],
            num_slot_tokens=template_slots.shape[1],
            num_template_tokens=template_tokens.shape[1],
            num_search_tokens=search_tokens.shape[1],
        )
        corrected_slots, correction_aux = self.semantic_slots.correct_slots(
            template_slots=template_slots,
            slot_candidate=slot_candidate, # 经过stage-3融合后的slot特征
            slot_attention=slot_attention,
        )
        enhanced_search_tokens, modulation_aux = self.semantic_slots.modulate_search(
            search_tokens=stage3_search_tokens,
            corrected_slots=corrected_slots,
        )

        dfs_state = dfs_state if isinstance(dfs_state, dict) else {}
        dfs_state["slot_state"] = correction_aux["slot_state"].detach()

        extras = {
            "slot_prior": slot_prior,
            "slot_template": template_slots,
            "slot_candidate": correction_aux["slot_candidate"],
            "slot_state": correction_aux["slot_state"],
            "slot_attention": correction_aux["slot_attention"],
            "slot_assignment": correction_aux["slot_assignment"],
            "slot_reliability": correction_aux["slot_reliability"],
            "slot_focus": correction_aux["slot_focus"],
            "slot_similarity": correction_aux["slot_similarity"],
            "text_slot_attention": init_aux["text_slot_attention"],
            "template_slot_attention": template_aux["template_slot_attention"],
            "search_slot_attention": modulation_aux["search_slot_attention"],
            "stage3_attn": stage3_aux["attn"],
            "stage3_backbone_feat": stage3_tokens,
            "stage3_template_feat": stage3_template_tokens,
            "stage3_search_feat": stage3_search_tokens,
        }
        return enhanced_search_tokens, extras, dfs_state

    def forward_text(self, captions, device):
        tokenized = self.tokenizer.batch_encode_plus(
            captions,
            padding="longest",
            return_tensors="pt",
        ).to(device)
        encoded_text = self.text_encoder(**tokenized)

        text_attention_mask = tokenized.attention_mask.ne(1).bool()
        text_features = self.text_adj(encoded_text.last_hidden_state)

        valid_mask = (~text_attention_mask).unsqueeze(-1).float()
        pooled_text = (text_features * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        return NestedTensor(text_features, text_attention_mask), pooled_text

    def forward(
        self,
        template,
        search,
        soft_token_template_mask=None,
        exp_str=None,
        cached_text_feat=None,
        dfs_state=None,
        training=True,
    ):
        if isinstance(search, (list, tuple)):
            if len(search) != 1:
                raise ValueError("DFSTrack forward only supports a single search image")
            search = search[0]

        b0 = template[0].shape[0]

        if training: # 返回的是文本特征以及掩码，_为池化后的文本特征
            text_features, _ = self.forward_text(exp_str, device=search.device)
        else:
            text_features = exp_str
            _ = cached_text_feat

        template_tokens_list, search_tokens = self.forward_backbone_enc12(
            template,
            search,
            soft_token_template_mask,
        )
        template_tokens = torch.cat(template_tokens_list, dim=1)
        if template_tokens.shape[1] != self.backbone.pos_embed_z.shape[1]:
            raise ValueError(
                "Template token count does not match backbone positional embeddings. "
                "Please keep cfg.DATA.TEMPLATE.NUMBER consistent with the actual number of templates."
            )

        enhanced_search_tokens = search_tokens
        extras = {}
        backbone_feat = None
        if self.dfs_enabled:
            enhanced_search_tokens, extras, dfs_state = self._run_dfs_branch(
                template_tokens=template_tokens,
                search_tokens=search_tokens,
                text_features=text_features,
                dfs_state=dfs_state,
            )
            backbone_feat = extras["stage3_backbone_feat"]
        elif not training:
            dfs_state = dfs_state if isinstance(dfs_state, dict) else {}
        else:
            x_pos = self._build_stage3_pos_embed(batch_size=b0, num_slot_tokens=0)
            backbone_feat, aux_dict = self.forward_backbone_stage3(
                token_groups=[template_tokens, enhanced_search_tokens],
                x_pos=x_pos,
                return_last_attn=False,
            )
            enhanced_search_tokens = backbone_feat[:, -self.feat_len_s:]

        if self.dfs_enabled:
            aux_dict = {"attn": extras["stage3_attn"]}
            search_tokens = enhanced_search_tokens
        else:
            search_tokens = enhanced_search_tokens

        opt_feat = search_tokens.transpose(1, 2).contiguous().view(
            -1,
            self.dim,
            self.feat_sz_s,
            self.feat_sz_s,
        )

        out = self.forward_head(opt_feat)
        out.update(aux_dict)
        out["backbone_feat"] = backbone_feat
        out["text_feat"] = text_features.tensors
        if self.dfs_enabled:
            out.update(extras)
        if not training:
            out["dfs_state"] = dfs_state
        return out

    def forward_head(self, opt_feat, gt_score_map=None):
        bs = opt_feat.shape[0]
        nq = 1

        if self.head_type == "CORNER":
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            return {
                "pred_boxes": outputs_coord.view(bs, nq, 4).contiguous(),
                "score_map": score_map,
            }

        if self.head_type == "CENTER":
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            return {
                "pred_boxes": bbox.view(bs, nq, 4).contiguous(),
                "score_map": score_map_ctr,
                "size_map": size_map,
                "offset_map": offset_map,
            }

        raise NotImplementedError


def build_dfstrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, "../../../resource/pretrained_models")
    text_pretrained_path = os.path.join(pretrained_path, "roberta-base")

    if cfg.MODEL.PRETRAIN_FILE and training and ("DFSTrack" not in cfg.MODEL.PRETRAIN_FILE):
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ""

    if cfg.MODEL.BACKBONE.TYPE == "itpn_base":
        backbone = fast_itpn_base_3324_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == "itpn_large":
        backbone = fast_itpn_large_2240_patch16_256(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, dim=hidden_dim, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)
    if os.path.isdir(text_pretrained_path):
        text_model_id = text_pretrained_path
    else:
        text_model_id = "roberta-base"
        print("Local RoBERTa weights not found, fallback to Hugging Face model hub:", text_model_id)

    tokenizer = RobertaTokenizerFast.from_pretrained(text_model_id)
    text_encoder = RobertaModel.from_pretrained(text_model_id)

    model = DFSTrack(
        backbone,
        box_head,
        tokenizer,
        text_encoder,
        head_type=cfg.MODEL.HEAD.TYPE,
        dim=hidden_dim,
        cfg=cfg,
    )

    if cfg.MODEL.PRETRAINED_PATH and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAINED_PATH, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print("Load pretrained model from: " + cfg.MODEL.PRETRAINED_PATH)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

    return model
