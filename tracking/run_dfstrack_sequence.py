import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "demo_results"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def natural_key(path: Path):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", path.name)]


def load_frame_paths(frames_dir: str):
    frame_dir = Path(frames_dir)
    if not frame_dir.is_dir():
        raise ValueError(f"frames_dir does not exist: {frames_dir}")

    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    frame_paths = [p for p in frame_dir.iterdir() if p.suffix.lower() in valid_suffixes]
    frame_paths = sorted(frame_paths, key=natural_key)

    if not frame_paths:
        raise ValueError(f"No image frames found in: {frames_dir}")

    return frame_paths


def parse_bbox(bbox_str: str):
    parts = [p for p in re.split(r"[,\s]+", bbox_str.strip()) if p]
    if len(parts) != 4:
        raise ValueError("init_bbox must contain 4 numbers in xywh format, e.g. '100,120,80,90'")

    bbox = [float(v) for v in parts]
    if bbox[2] <= 0 or bbox[3] <= 0:
        raise ValueError(f"Invalid init_bbox with non-positive size: {bbox}")
    return bbox


def resolve_config_path(config_arg: str):
    config_path = Path(config_arg)
    if config_path.is_file():
        return config_path

    yaml_path = PROJECT_ROOT / "experiments" / "dfstrack" / f"{config_arg}.yaml"
    if yaml_path.is_file():
        return yaml_path

    raise ValueError(
        f"Cannot find config file from '{config_arg}'. "
        f"Expected an existing yaml path or experiments/dfstrack/{config_arg}.yaml"
    )


def build_params(config_path: Path, checkpoint_path: str):
    from lib.config.dfstrack.config import cfg, update_config_from_file
    from lib.test.utils import TrackerParams

    update_config_from_file(str(config_path))

    params = TrackerParams()
    params.cfg = cfg
    params.yaml_name = config_path.stem
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.checkpoint = checkpoint_path
    params.save_all_boxes = False
    return params


def read_first_bbox_from_txt(txt_path: Path):
    bbox_series = read_bbox_series_from_txt(txt_path)
    if not bbox_series:
        raise ValueError(f"No bbox annotation found in {txt_path}")
    first_bbox = bbox_series[0]
    if first_bbox[2] <= 0 or first_bbox[3] <= 0:
        raise ValueError(f"Invalid initial bbox in {txt_path}: {first_bbox}")
    return first_bbox


def read_bbox_series_from_txt(txt_path: Path):
    bbox_series = []
    with txt_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = [p for p in re.split(r"[,\s\t]+", line) if p]
            if len(parts) < 4:
                raise ValueError(f"Invalid bbox line in {txt_path}: {line}")
            bbox = [float(parts[i]) for i in range(4)]
            bbox_series.append(bbox)
    return bbox_series


def read_text_prompt(txt_path: Path):
    if not txt_path.is_file():
        return None
    text = txt_path.read_text(encoding="utf-8").strip()
    return text or None


def get_demo_env_settings():
    local_py = PROJECT_ROOT / "lib" / "test" / "evaluation" / "local.py"
    if not local_py.is_file():
        raise ValueError(f"Cannot find local.py at: {local_py}")

    settings = {}
    pattern = re.compile(r"""settings\.(\w+)\s*=\s*(['"])(.*?)\2""")
    with local_py.open("r", encoding="utf-8") as f:
        for raw_line in f:
            match = pattern.search(raw_line)
            if match:
                settings[match.group(1)] = match.group(3)
    return settings


def get_demo_dataset_base_path(dataset_name: str):
    settings = get_demo_env_settings()
    dataset_key = dataset_name.lower()

    dataset_to_attr = {
        "tnl2k": "tnl2k_path",
        "lasot_lang": "lasotlang_path",
        "lasot": "lasot_path",
    }
    if dataset_key not in dataset_to_attr:
        raise ValueError(
            f"Unsupported dataset_name='{dataset_name}'. "
            "Demo mode currently supports: tnl2k, lasot_lang, lasot."
        )

    base_path = settings.get(dataset_to_attr[dataset_key], "")
    if not base_path and dataset_key == "lasot":
        base_path = settings.get("lasotlang_path", "")
    if not base_path:
        raise ValueError(
            f"Path for dataset '{dataset_name}' is empty in lib/test/evaluation/local.py."
        )

    base_path = Path(base_path)
    if not base_path.is_dir():
        raise ValueError(
            f"Dataset path for '{dataset_name}' does not exist: {base_path}"
        )
    return base_path


def load_dataset_sequence(dataset_name: str, sequence_name: str):
    dataset_key = dataset_name.lower()
    base_path = get_demo_dataset_base_path(dataset_name)
    sequence_names = list_dataset_sequences(dataset_name)

    if sequence_name not in sequence_names:
        matched = [name for name in sequence_names if sequence_name.lower() in name.lower()]
        hint = ""
        if matched:
            hint = f" Similar names: {matched[:10]}"
        raise ValueError(
            f"Sequence '{sequence_name}' was not found in dataset '{dataset_name}'.{hint}"
        )

    if dataset_key == "tnl2k":
        sequence_dir = base_path / sequence_name
        return {
            "name": sequence_name,
            "frame_paths": load_frame_paths(str(sequence_dir / "imgs")),
            "init_bbox": read_first_bbox_from_txt(sequence_dir / "groundtruth.txt"),
            "init_text": read_text_prompt(sequence_dir / "language.txt"),
            "gt_bboxes": read_bbox_series_from_txt(sequence_dir / "groundtruth.txt"),
        }

    class_name = sequence_name.split("-")[0]
    sequence_dir = base_path / class_name / sequence_name
    init_text = None
    if dataset_key == "lasot_lang":
        init_text = read_text_prompt(sequence_dir / "nlp.txt")

    return {
        "name": sequence_name,
        "frame_paths": load_frame_paths(str(sequence_dir / "img")),
        "init_bbox": read_first_bbox_from_txt(sequence_dir / "groundtruth.txt"),
        "init_text": init_text,
        "gt_bboxes": read_bbox_series_from_txt(sequence_dir / "groundtruth.txt"),
    }


def list_dataset_sequences(dataset_name: str):
    dataset_key = dataset_name.lower()
    base_path = get_demo_dataset_base_path(dataset_name)

    if dataset_key == "tnl2k":
        return sorted([path.name for path in base_path.iterdir() if path.is_dir()])

    if dataset_key in {"lasot_lang", "lasot"}:
        sequence_names = []
        for class_dir in sorted([path for path in base_path.iterdir() if path.is_dir()]):
            for seq_dir in sorted([path for path in class_dir.iterdir() if path.is_dir()]):
                sequence_names.append(seq_dir.name)
        return sequence_names

    raise ValueError(
        f"Unsupported dataset_name='{dataset_name}'. Demo mode currently supports: tnl2k, lasot_lang, lasot."
    )


def ensure_text_prompt(init_text, dataset_name: str, sequence_name: str):
    if init_text is None:
        raise ValueError(
            f"Sequence '{sequence_name}' from dataset '{dataset_name}' does not provide a language query. "
            "Please pass --init_text explicitly for the demo."
        )

    init_text = str(init_text).strip()
    if not init_text:
        raise ValueError(
            f"Empty text prompt for sequence '{sequence_name}' from dataset '{dataset_name}'. "
            "Please provide a non-empty --init_text."
        )
    return init_text


def resolve_demo_inputs(args):
    if args.sequence_name:
        seq = load_dataset_sequence(args.dataset_name, args.sequence_name)
        frame_paths = seq["frame_paths"]
        init_bbox = seq["init_bbox"]
        if init_bbox is None:
            raise ValueError(
                f"Sequence '{seq['name']}' from dataset '{args.dataset_name}' does not contain an initial bbox."
            )
        init_text = args.init_text if args.init_text is not None else seq["init_text"]
        init_text = ensure_text_prompt(init_text, args.dataset_name, seq["name"])
        return {
            "frame_paths": frame_paths,
            "init_bbox": [float(v) for v in init_bbox],
            "init_text": init_text,
            "sequence_name": seq["name"],
            "dataset_name": args.dataset_name,
            "source": "dataset",
            "gt_bboxes": seq.get("gt_bboxes"),
        }

    if args.frames_dir is None:
        raise ValueError("Manual demo mode requires --frames_dir when --sequence_name is not used.")
    if args.init_bbox is None:
        raise ValueError("Manual demo mode requires --init_bbox when --sequence_name is not used.")
    if args.init_text is None:
        raise ValueError("Manual demo mode requires --init_text when --sequence_name is not used.")

    frame_paths = load_frame_paths(args.frames_dir)
    init_bbox = parse_bbox(args.init_bbox)
    init_text = ensure_text_prompt(args.init_text, args.dataset_name, Path(args.frames_dir).name)
    gt_bboxes = None
    if args.gt_file is not None:
        gt_bboxes = read_bbox_series_from_txt(Path(args.gt_file))
    return {
        "frame_paths": frame_paths,
        "init_bbox": init_bbox,
        "init_text": init_text,
        "sequence_name": Path(args.frames_dir).name,
        "dataset_name": args.dataset_name,
        "source": "manual",
        "gt_bboxes": gt_bboxes,
    }


def resolve_output_dir(output_dir_arg: str, dataset_name: str, sequence_name: str):
    output_dir = Path(output_dir_arg)
    try:
        is_default_root = output_dir.resolve() == DEFAULT_OUTPUT_ROOT.resolve()
    except Exception:
        is_default_root = output_dir == DEFAULT_OUTPUT_ROOT

    if is_default_root:
        return output_dir / dataset_name / sequence_name
    return output_dir


def calc_iou_xywh(pred_bbox, gt_bbox):
    if pred_bbox[2] <= 0 or pred_bbox[3] <= 0 or gt_bbox[2] <= 0 or gt_bbox[3] <= 0:
        return -1.0

    x1 = max(pred_bbox[0], gt_bbox[0])
    y1 = max(pred_bbox[1], gt_bbox[1])
    x2 = min(pred_bbox[0] + pred_bbox[2] - 1.0, gt_bbox[0] + gt_bbox[2] - 1.0)
    y2 = min(pred_bbox[1] + pred_bbox[3] - 1.0, gt_bbox[1] + gt_bbox[3] - 1.0)

    inter_w = max(0.0, x2 - x1 + 1.0)
    inter_h = max(0.0, y2 - y1 + 1.0)
    intersection = inter_w * inter_h
    union = pred_bbox[2] * pred_bbox[3] + gt_bbox[2] * gt_bbox[3] - intersection
    if union <= 0.0:
        return -1.0
    return intersection / union


def compute_frame_ious(predictions, gt_bboxes):
    iou_values = []
    for frame_idx, pred in enumerate(predictions):
        if frame_idx >= len(gt_bboxes):
            iou_values.append(-1.0)
            continue
        iou_values.append(calc_iou_xywh(pred["target_bbox"], gt_bboxes[frame_idx]))
    return iou_values


def draw_box(frame, bbox, score=None):
    x, y, w, h = [int(round(v)) for v in bbox]
    x2 = x + w
    y2 = y + h
    vis_frame = frame.copy()
    cv2.rectangle(vis_frame, (x, y), (x2, y2), (0, 255, 0), 2)

    if score is not None:
        score_text = f"score={score:.4f}"
        cv2.putText(
            vis_frame,
            score_text,
            (max(5, x), max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return vis_frame


def save_plain_bboxes(save_path: Path, predictions):
    with save_path.open("w", encoding="utf-8") as f:
        for pred in predictions:
            bbox = pred["target_bbox"]
            f.write(",".join(f"{value:.4f}" for value in bbox) + "\n")


def save_scalar_series(save_path: Path, values):
    with save_path.open("w", encoding="utf-8") as f:
        for value in values:
            f.write(f"{float(value):.6f}\n")


def run_sequence(args):
    if not args.checkpoint:
        raise ValueError("Running demo inference requires --checkpoint.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "DFSTRACK test inference currently requires CUDA because the tracker/preprocessor uses GPU tensors."
        )

    from lib.test.tracker.dfstrack import DFSTRACK

    demo_inputs = resolve_demo_inputs(args)
    frame_paths = demo_inputs["frame_paths"]
    init_bbox = demo_inputs["init_bbox"]
    init_text = demo_inputs["init_text"]
    sequence_name = demo_inputs["sequence_name"]
    dataset_name = demo_inputs["dataset_name"]
    gt_bboxes = demo_inputs.get("gt_bboxes")
    config_path = resolve_config_path(args.config)
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.is_file():
        raise ValueError(f"Checkpoint does not exist: {args.checkpoint}")

    params = build_params(config_path, str(checkpoint_path))
    tracker = DFSTRACK(params, dataset_name=dataset_name)

    output_dir = resolve_output_dir(args.output_dir, dataset_name, sequence_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise ValueError(f"Failed to read the first frame: {frame_paths[0]}")

    init_info = {
        "init_bbox": init_bbox,
        "init_nlp": init_text,
        "seq_name": sequence_name,
    }
    tracker.initialize(first_frame, init_info)

    predictions = [
        {
            "frame_idx": 0,
            "frame_name": frame_paths[0].name,
            "target_bbox": [float(v) for v in init_bbox],
            "best_score": 1.0,
        }
    ]

    video_writer = None
    if args.save_video:
        video_path = output_dir / "tracking.mp4"
        h, w = first_frame.shape[:2]
        video_writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (w, h),
        )
        video_writer.write(draw_box(first_frame, init_bbox, score=1.0))

    prev_output = {"target_bbox": init_bbox, "best_score": 1.0}

    for frame_idx, frame_path in enumerate(frame_paths[1:], start=1):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise ValueError(f"Failed to read frame: {frame_path}")

        track_info = {
            "seq_name": sequence_name,
            "previous_output": prev_output,
        }
        out = tracker.track(frame, track_info)

        pred_bbox = [float(v) for v in out["target_bbox"]]
        best_score = float(out["best_score"])
        pred_record = {
            "frame_idx": frame_idx,
            "frame_name": frame_path.name,
            "target_bbox": pred_bbox,
            "best_score": best_score,
        }
        predictions.append(pred_record)
        prev_output = out

        if video_writer is not None:
            video_writer.write(draw_box(frame, pred_bbox, score=best_score))

    if video_writer is not None:
        video_writer.release()

    json_path = output_dir / "predictions.json"
    txt_path = output_dir / "target_bbox.txt"
    iou_path = output_dir / "target_iou.txt"
    iou_values = None
    if gt_bboxes is not None:
        iou_values = compute_frame_ious(predictions, gt_bboxes)
        if len(gt_bboxes) != len(predictions):
            print(
                f"Warning: gt length ({len(gt_bboxes)}) != prediction length ({len(predictions)}). "
                "Missing GT frames are written as -1.000000 in target_iou.txt."
            )

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": str(config_path),
                "checkpoint": str(checkpoint_path),
                "frames_dir": str(Path(frame_paths[0]).parent.resolve()),
                "dataset_name": dataset_name,
                "sequence_name": sequence_name,
                "input_source": demo_inputs["source"],
                "init_bbox": init_bbox,
                "init_text": init_text,
                "gt_available": gt_bboxes is not None,
                "predictions": predictions,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    save_plain_bboxes(txt_path, predictions)
    if iou_values is not None:
        save_scalar_series(iou_path, iou_values)

    print(f"Sequence: {sequence_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Frames: {len(frame_paths)}")
    print(f"Results saved to: {output_dir}")
    print(f"BBoxes txt: {txt_path}")
    if iou_values is not None:
        print(f"IoU txt: {iou_path}")
    else:
        print("IoU txt: skipped (no GT provided)")
    print(f"Predictions json: {json_path}")
    if args.save_video:
        print(f"Visualization video: {output_dir / 'tracking.mp4'}")


def build_argparser():
    parser = argparse.ArgumentParser(
        description=(
            "Run DFSTRACK on a single image sequence. "
            "You can either provide a raw frames_dir, or provide dataset_name + sequence_name "
            "to demo a registered dataset sequence directly."
        )
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default=None,
        help="Manual mode only. Directory containing sequence frames, sorted by file name.",
    )
    parser.add_argument(
        "--sequence_name",
        type=str,
        default=None,
        help="Dataset demo mode only. Exact sequence name inside the selected dataset.",
    )
    parser.add_argument(
        "--list_sequences",
        action="store_true",
        help="List all available sequence names for --dataset_name and exit.",
    )
    parser.add_argument(
        "--init_bbox",
        type=str,
        default=None,
        help="Manual mode only. Initial target bbox in xywh format, e.g. '120,85,64,96'.",
    )
    parser.add_argument(
        "--init_text",
        type=str,
        default=None,
        help=(
            "Language prompt for initialization. "
            "If sequence_name mode is used, dataset text will be used by default when available."
        ),
    )
    parser.add_argument(
        "--gt_file",
        type=str,
        default=None,
        help=(
            "Manual mode only. Optional GT bbox txt in xywh per line. "
            "If provided, the script will also save per-frame IoU to target_iou.txt."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/ssa/code/DFSTrack/output/checkpoints/train/dftrack/dftrack_base/DFSTrack_ep0174.pth.tar",
        help="Path to the DFSTRACK checkpoint (.pth.tar).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="dfstrack_base",
        help="Yaml path or config name under experiments/dfstrack.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="demo",
        help=(
            "In dataset demo mode, this selects the registered dataset, e.g. tnl2k / lasot_lang / lasot. "
            "In manual mode, it is passed to the tracker only for test-time update rules."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help=(
            "Directory to save predicted boxes and optional visualization video. "
            "If left as default, outputs go to tracking/demo_results/<dataset>/<sequence>/."
        ),
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="If set, save an mp4 visualization with predicted boxes.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS for the saved visualization video.",
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    args.dataset_name = "tnl2k"
    args.sequence_name = "NBA_Bull_video_02-Done"


    if args.list_sequences: # 查看数据集中的序列列表
        if args.dataset_name == "demo":
            raise ValueError("--list_sequences requires a real --dataset_name, e.g. tnl2k or lasot_lang.")
        sequence_names = list_dataset_sequences(args.dataset_name)
        print(f"Dataset: {args.dataset_name}")
        print(f"Num sequences: {len(sequence_names)}")
        for sequence_name in sequence_names:
            print(sequence_name)
        return
    

    run_sequence(args)


if __name__ == "__main__":
    """
    支持手动模式：
    python tracking/run_dfstrack_sequence.py \
        --frames_dir /path/to/frames \
        --init_bbox "120,85,64,96" \
        --init_text "the target object" \
        --gt_file /path/to/groundtruth.txt \
        --checkpoint output/checkpoints/train/dftrack/dftrack_base/DFSTrack_ep0174.pth.tar
    也支持跑数据集中的序列（以tnl2k为例）：、
    python tracking/run_dfstrack_sequence.py \
        --dataset_name tnl2k \
        --sequence_name basketball1-1 \
        --checkpoint output/checkpoints/train/dftrack/dftrack_base/DFSTrack_ep0174.pth.tar
     输出会保存在 /demo_results/tnl2k/basketball1-1/ 目录下。
     该脚本会保存预测的bbox txt和json文件，如果提供了GT还会保存IoU txt。也可以选择保存带可视化框的视频。
     注意：数据集模式会自动读取数据集中的文本提示，如果需要覆盖请使用 --init_text 参数。
    """
    main()
