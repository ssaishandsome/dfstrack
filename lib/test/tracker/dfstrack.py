from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.dfstrack_utils import sample_target
from lib.utils.box_ops import clip_box
from lib.models.dfstrack import build_dfstrack
from lib.test.tracker.dfstrack_utils import Preprocessor
from lib.test.utils.hann import hann2d
from lib.utils.ce_utils import generate_bbox_mask


def get_resize_template_bbox(template_bbox, resize_factor):
    w, h = template_bbox[2], template_bbox[3]
    w_1, h_1 = int(w * resize_factor), int(h * resize_factor)
    xc, yc = 64, 64
    x0, y0 = int(xc - w_1 * 0.5), int(yc - h_1 * 0.5)
    return [x0, y0, w_1, h_1]


class DFSTRACK(BaseTracker):
    def __init__(self, params, dataset_name):
        super().__init__(params)
        network = build_dfstrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location="cpu")["net"], strict=True)
        print("load from ", self.params.checkpoint)

        self.cfg = params.cfg
        self.num_template = self.cfg.TEST.NUM_TEMPLATES
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.frame_id = 0
        self.dataset_name = dataset_name

        dataset_name_upper = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, dataset_name_upper):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[dataset_name_upper]
        else:
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS.DEFAULT
        if hasattr(self.cfg.TEST.UPDATE_THRESHOLD, dataset_name_upper):
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD[dataset_name_upper]
        else:
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD.DEFAULT
        self.update_edge = 1e6

    def initialize(self, image, info: dict):
        z_patch_arr, resize_factor = sample_target(
            image,
            info["init_bbox"],
            self.params.template_factor,
            output_sz=self.params.template_size,
        )
        template = self.preprocessor.process(z_patch_arr)
        self.template_list = [template] * self.num_template

        template_bbox = info["init_bbox"]
        resize_template_bbox = get_resize_template_bbox(template_bbox, resize_factor)
        resize_template_bbox = [torch.tensor(resize_template_bbox).to(template.device)]
        bbox_mask = torch.zeros((1, self.params.template_size, self.params.template_size))
        bbox_mask = generate_bbox_mask(bbox_mask, resize_template_bbox)
        bbox_mask = bbox_mask.unfold(1, 16, 16).unfold(2, 16, 16)
        bbox_mask = bbox_mask.mean(dim=(-1, -2)).view(bbox_mask.shape[0], -1).unsqueeze(-1)
        bbox_mask = bbox_mask.to(template.device)
        self.soft_token_template_mask = [bbox_mask, bbox_mask]

        self.text_features, self.text_sentence_features = self.network.forward_text(
            [info["init_nlp"]],
            num_search=1,
            device=template.device,
        )

        self.state = info["init_bbox"]
        self.frame_id = 0

    def track(self, image, info: dict = None):
        h, w, _ = image.shape
        self.frame_id += 1

        x_patch_arr, resize_factor = sample_target(
            image,
            self.state,
            self.params.search_factor,
            output_sz=self.params.search_size,
        )
        search = self.preprocessor.process(x_patch_arr)

        with torch.no_grad():
            out_dict = self.network(
                self.template_list,
                search,
                self.soft_token_template_mask,
                exp_str=self.text_features,
                exp_subject_mask=self.text_sentence_features,
                training=False,
            )

        pred_score_map = out_dict["score_map"]
        response = self.output_window * pred_score_map
        pred_boxes, best_score = self.network.box_head.cal_bbox(
            response,
            out_dict["size_map"],
            out_dict["offset_map"],
            return_score=True,
        )
        conf_score = best_score[0][0].item()
        pred_box = (pred_boxes.view(-1, 4).mean(dim=0) * self.params.search_size / resize_factor).tolist()
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), h, w, margin=10)

        if self.num_template > 1 and self.frame_id < self.update_edge:
            if (self.frame_id % self.update_intervals == 0) and (conf_score > self.update_threshold):
                z_patch_arr, resize_factor = sample_target(
                    image,
                    self.state,
                    self.params.template_factor,
                    output_sz=self.params.template_size,
                )
                template = self.preprocessor.process(z_patch_arr)
                self.template_list.append(template)
                if len(self.template_list) > self.num_template:
                    self.template_list.pop(1)

                template_bbox = self.state
                resize_template_bbox = get_resize_template_bbox(template_bbox, resize_factor)
                resize_template_bbox = [torch.tensor(resize_template_bbox).to(template.device)]
                bbox_mask = torch.zeros((1, self.params.template_size, self.params.template_size))
                bbox_mask = generate_bbox_mask(bbox_mask, resize_template_bbox)
                bbox_mask = bbox_mask.unfold(1, 16, 16).unfold(2, 16, 16)
                bbox_mask = bbox_mask.mean(dim=(-1, -2)).view(bbox_mask.shape[0], -1).unsqueeze(-1)
                bbox_mask = bbox_mask.to(template.device)

                self.soft_token_template_mask.append(bbox_mask)
                if len(self.soft_token_template_mask) > self.num_template:
                    self.soft_token_template_mask.pop(1)

        return {"target_bbox": self.state, "best_score": conf_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        cx, cy, bw, bh = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * bw, cy_real - 0.5 * bh, bw, bh]


def get_tracker_class():
    return DFSTRACK
