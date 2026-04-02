from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.heapmap_utils import generate_heatmap
from lib.utils.ce_utils import generate_bbox_mask
from lib.train.admin import multigpu


class DFSTrackActor(BaseActor):
    """Actor for training the clean DFSTrack baseline."""

    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.cfg = cfg

    def fix_bns(self):
        net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
        net.box_head.apply(self.fix_bn)

    def fix_bn(self, module):
        classname = module.__class__.__name__
        if classname.find("BatchNorm") != -1:
            module.eval()

    def __call__(self, data):
        out_dict = self.forward_pass(data)
        loss, status = self.compute_losses(out_dict, data)
        return loss, status

    def forward_pass(self, data):
        template_list, search_list = [], []
        for i in range(self.settings.num_template):
            template_img_i = data["template_images"][i].view(-1, *data["template_images"].shape[2:])
            template_list.append(template_img_i)

        for i in range(self.settings.num_search):
            search_img_i = data["search_images"][i].view(-1, *data["search_images"].shape[2:])
            search_list.append(search_img_i)

        bbox_mask_list = []
        for template_item in data["template_anno"]:
            template_bbox = template_item * template_list[0].shape[2]
            bbox_mask = torch.zeros( # b`s * 3 * 128 * 128
                (template_list[0].shape[0], template_list[0].shape[2], template_list[0].shape[3]),
                device=template_list[0].device,
            ) # 根据边界框对模板生成掩码，边界框内为1，边界框外为0
            bbox_mask = generate_bbox_mask(bbox_mask, template_bbox)
            bbox_mask = bbox_mask.unfold(1, 16, 16).unfold(2, 16, 16) # 只是为了生成和特征图大小一样的掩码，16是因为特征图是128/8=16
            bbox_mask = bbox_mask.mean(dim=(-1, -2)).view(bbox_mask.shape[0], -1).unsqueeze(-1)
            bbox_mask_list.append(bbox_mask)

        exp_str_list = []
        for item in data["nlp"]:
            exp_str_list.append(item.split("+")[0])

        out_dict = self.net(
            template=template_list,
            search=search_list,
            soft_token_template_mask=bbox_mask_list,
            exp_str=exp_str_list, # language
        )

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        gt_bbox = gt_dict["search_anno"].view(-1, 4)
        gts = gt_bbox.unsqueeze(0)
        gt_gaussian_maps = generate_heatmap(gts, self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        pred_boxes = pred_dict["pred_boxes"]
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")

        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
        gt_boxes_vec = (
            box_xywh_to_xyxy(gt_bbox)[:, None, :]
            .repeat((1, num_queries, 1))
            .view(-1, 4)
            .clamp(min=0.0, max=1.0)
        )

        try:
            giou_loss, iou = self.objective["giou"](pred_boxes_vec, gt_boxes_vec)
        except Exception:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        l1_loss = self.objective["l1"](pred_boxes_vec, gt_boxes_vec)
        location_loss = self.objective["focal"](pred_dict["score_map"], gt_gaussian_maps)

        loss = (
            self.loss_weight["giou"] * giou_loss
            + self.loss_weight["l1"] * l1_loss
            + self.loss_weight["focal"] * location_loss
        )

        if return_status:
            mean_iou = iou.detach().mean()
            status = {
                "Loss/total": loss.item(),
                "Loss/giou": giou_loss.item(),
                "Loss/l1": l1_loss.item(),
                "Loss/location": location_loss.item(),
                "IoU_main": mean_iou.item(),
            }
            return loss, status

        return loss
