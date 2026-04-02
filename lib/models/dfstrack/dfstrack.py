import os

import torch
from torch import nn

from transformers import RobertaModel, RobertaTokenizerFast

from lib.models.backbones.fast_itpn import (
    fast_itpn_base_3324_patch16_224,
    fast_itpn_large_2240_patch16_256,
)
from lib.models.layers.head import build_box_head
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.misc import NestedTensor


class DFSTrack(nn.Module):
    """A clean visual-language tracking baseline with shared visual/text encoders."""

    def __init__(self, backbone, box_head, tokenizer, text_encoder, head_type="CENTER", dim=512):
        super().__init__()
        self.backbone = backbone
        self.box_head = box_head
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.head_type = head_type
        self.dim = dim

        if head_type in ["CORNER", "CENTER"]:
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        self.cls_prompts_pos = nn.Embedding(num_embeddings=1, embedding_dim=self.dim)
        self.text_adj = nn.Sequential(
            nn.Linear(768, self.dim, bias=True),
            nn.LayerNorm(self.dim, eps=1e-12),
            nn.Dropout(0.1),
        )
        self.text_gate = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
            nn.Sigmoid(),
        )

    def forward_backbone(self, template, search, soft_token_template_mask, x_pos):
        template = [template[:, :3], template[:, 3:]]
        soft_token_template_mask = [
            soft_token_template_mask[:, :64],
            soft_token_template_mask[:, 64:],
        ]

        x, _ = self.backbone.forward_features_pe(
            z=template,
            x=search,
            soft_token_template_mask=soft_token_template_mask,
        )
        x, aux_dict = self.backbone.forward_features_stage3(x, None, x_pos)
        return x, aux_dict

    def forward_text(self, captions, num_search, device):
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

        text_features_t = []
        text_attention_mask_t = []
        pooled_text_t = []
        for _ in range(num_search):
            text_features_t.append(text_features)
            text_attention_mask_t.append(text_attention_mask)
            pooled_text_t.append(pooled_text)

        text_features = NestedTensor(torch.cat(text_features_t, dim=0), torch.cat(text_attention_mask_t, dim=0))
        pooled_text = torch.cat(pooled_text_t, dim=0)
        return text_features, pooled_text

    def forward(
        self,
        template,
        search,
        soft_token_template_mask=None,
        exp_str=None,
        exp_subject_mask=None,
        temporal_infor=None,
        first_frame_flag=False,
        training=True,
    ):
        del temporal_infor, first_frame_flag

        b0, num_search = template[0].shape[0], len(search)
        if training:
            search = torch.cat(search, dim=0)
            template = torch.cat(template, dim=1)
            soft_token_template_mask = torch.cat(soft_token_template_mask, dim=1)

            template_temporal = []
            soft_token_template_mask_temporal = []
            for _ in range(num_search):
                template_temporal.append(template)
                soft_token_template_mask_temporal.append(soft_token_template_mask)
            template_temporal = torch.cat(template_temporal, dim=0)
            soft_token_template_mask_temporal = torch.cat(soft_token_template_mask_temporal, dim=0)
            # text_sentence_features 是每个文本的全局特征，text_features 是每个文本的token级别特征
            text_features, text_sentence_features = self.forward_text(exp_str, num_search, device=search.device)
        else:
            b0 = 1
            template_temporal = torch.cat(template, dim=1)
            soft_token_template_mask_temporal = torch.cat(soft_token_template_mask, dim=1)
            text_features = exp_str
            text_sentence_features = exp_subject_mask

        cls_prompts_pos = self.cls_prompts_pos.weight.unsqueeze(0)
        x_pos_0 = torch.cat([cls_prompts_pos, self.backbone.pos_embed_z, self.backbone.pos_embed_x], dim=1)
        x_pos = x_pos_0.repeat(b0 * num_search, 1, 1)

        x, aux_dict = self.forward_backbone(
            template_temporal,
            search,
            soft_token_template_mask_temporal,
            x_pos,
        )

        search_tokens = x[:, -self.feat_len_s:]
        # 句级文本向量去调制 search tokens
        text_gate = self.text_gate(text_sentence_features).unsqueeze(1)
        conditioned_search_tokens = search_tokens + search_tokens * text_gate + text_sentence_features.unsqueeze(1)

        opt_feat = conditioned_search_tokens.transpose(1, 2).contiguous().view(
            -1,
            self.dim,
            self.feat_sz_s,
            self.feat_sz_s,
        )

        out = self.forward_head(opt_feat)
        out.update(aux_dict)
        out["backbone_feat"] = x
        out["text_feat"] = text_features.tensors
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
    )

    if cfg.MODEL.PRETRAINED_PATH and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAINED_PATH, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print("Load pretrained model from: " + cfg.MODEL.PRETRAINED_PATH)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

    return model
