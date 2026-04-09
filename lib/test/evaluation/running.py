import numpy as np
import multiprocessing
import os
import sys
from itertools import product
from collections import OrderedDict
from lib.test.evaluation import Sequence, Tracker
import torch


def _calc_sequence_iou(pred_bb, anno_bb, target_visible=None):
    """Compute per-frame IoU for single-object tracking results.

    Args:
        pred_bb: [T, 4] predicted boxes in xywh image coordinates.
        anno_bb: [T, 4] ground-truth boxes in xywh image coordinates.
        target_visible: optional [T] visibility flags.

    Returns:
        np.ndarray: [T] IoU values. Invalid frames are marked as -1.
    """
    pred_bb = np.asarray(pred_bb, dtype=np.float64)
    anno_bb = np.asarray(anno_bb, dtype=np.float64)

    if pred_bb.ndim == 1:
        pred_bb = pred_bb.reshape(1, 4)
    if anno_bb.ndim == 1:
        anno_bb = anno_bb.reshape(1, 4)

    seq_len = min(pred_bb.shape[0], anno_bb.shape[0])
    pred_bb = pred_bb[:seq_len]
    anno_bb = anno_bb[:seq_len]

    iou = np.full(seq_len, -1.0, dtype=np.float64)

    valid_gt = (anno_bb[:, 2] > 0.0) & (anno_bb[:, 3] > 0.0)
    valid_pred = (pred_bb[:, 2] > 0.0) & (pred_bb[:, 3] > 0.0)
    valid = valid_gt & valid_pred

    if target_visible is not None:
        target_visible = np.asarray(target_visible).astype(np.bool_)[:seq_len]
        valid = valid & target_visible

    tl = np.maximum(pred_bb[:, :2], anno_bb[:, :2])
    br = np.minimum(pred_bb[:, :2] + pred_bb[:, 2:] - 1.0, anno_bb[:, :2] + anno_bb[:, 2:] - 1.0)
    wh = np.clip(br - tl + 1.0, a_min=0.0, a_max=None)

    intersection = wh[:, 0] * wh[:, 1]
    union = pred_bb[:, 2] * pred_bb[:, 3] + anno_bb[:, 2] * anno_bb[:, 3] - intersection

    valid = valid & (union > 0.0)
    iou[valid] = intersection[valid] / union[valid]
    return iou


def _save_tracker_output(seq: Sequence, tracker: Tracker, output: dict):
    """Saves the output of the tracker."""

    if not os.path.exists(tracker.results_dir):
        print("create tracking result dir:", tracker.results_dir)
        os.makedirs(tracker.results_dir)
    # tracking/test.py already sets tracker.results_dir to:
    #   env.save_dir / tracker_param / dataset_name
    # Keep one dataset-level directory only, e.g. dfstrack_base/tnl2k/<seq>.txt.
    base_results_path = os.path.join(tracker.results_dir, seq.name)

    def save_bb(file, data):
        tracked_bb = np.array(data).astype(int)
        np.savetxt(file, tracked_bb, delimiter='\t', fmt='%d')

    def save_time(file, data):
        exec_times = np.array(data).astype(float)
        np.savetxt(file, exec_times, delimiter='\t', fmt='%f')

    def save_score(file, data):
        scores = np.array(data).astype(float)
        np.savetxt(file, scores, delimiter='\t', fmt='%.2f')

    def save_iou(file, data):
        ious = np.array(data).astype(float)
        np.savetxt(file, ious, delimiter='\t', fmt='%.6f')

    def _convert_dict(input_dict):
        data_dict = {}
        for elem in input_dict:
            for k, v in elem.items():
                if k in data_dict.keys():
                    data_dict[k].append(v)
                else:
                    data_dict[k] = [v, ]
        return data_dict

    for key, data in output.items():
        # If data is empty
        if not data:
            continue

        if key == 'target_bbox':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        if key == 'all_boxes':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_all_boxes.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}_all_boxes.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        if key == 'all_scores':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_all_scores.txt'.format(base_results_path, obj_id)
                    save_score(bbox_file, d)
            else:
                # Single-object mode
                print("saving scores...")
                bbox_file = '{}_all_scores.txt'.format(base_results_path)
                save_score(bbox_file, data)

        elif key == 'time':
            if isinstance(data[0], dict):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    timings_file = '{}_{}_time.txt'.format(base_results_path, obj_id)
                    save_time(timings_file, d)
            else:
                timings_file = '{}_time.txt'.format(base_results_path)
                save_time(timings_file, data)

    # Save per-frame IoU when GT is available.
    # Keep time.txt unchanged so the existing FPS/statistics flow is not broken.
    if seq.object_ids is None and 'target_bbox' in output and output['target_bbox'] and seq.ground_truth_rect is not None:
        if not isinstance(output['target_bbox'][0], (dict, OrderedDict)):
            iou_file = '{}_iou.txt'.format(base_results_path)
            target_visible = getattr(seq, 'target_visible', None)
            iou_values = _calc_sequence_iou(output['target_bbox'], seq.ground_truth_rect, target_visible)
            save_iou(iou_file, iou_values)


def run_sequence(seq: Sequence, tracker: Tracker, debug=False, num_gpu=8):
    """Runs a tracker on a sequence."""
    '''2021.1.2 Add multiple gpu support'''
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    def _results_exist():
        if seq.object_ids is None:
            base_results_path = os.path.join(tracker.results_dir, seq.name)
            bbox_file = '{}.txt'.format(base_results_path)

            if not os.path.isfile(bbox_file):
                return False

            # Re-run if GT is available but the per-frame IoU file has not been generated yet.
            if seq.ground_truth_rect is not None:
                iou_file = '{}_iou.txt'.format(base_results_path)
                if not os.path.isfile(iou_file):
                    return False

            return True
        else:
            bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq.name, obj_id) for obj_id in seq.object_ids]
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0

    if _results_exist() and not debug:
        print('FPS: {}'.format(-1))
        return

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

    output = tracker.run_sequence(seq, debug=debug)
    # if debug:
    #     output = tracker.run_sequence(seq, debug=debug)
    # else:
    #     try:
    #         output = tracker.run_sequence(seq, debug=debug)
    #     except Exception as e:
    #         print(e)
    #         return

    sys.stdout.flush()

    if isinstance(output['time'][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output['time']])
        num_frames = len(output['time'])
    else:
        exec_time = sum(output['time'])
        num_frames = len(output['time'])

    print('FPS: {}'.format(num_frames / exec_time))

    if not debug:
        _save_tracker_output(seq, tracker, output)


def run_dataset(dataset, trackers, debug=False, threads=0, num_gpus=8):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
    """
    multiprocessing.set_start_method('spawn', force=True)

    print('Evaluating {:4d} trackers on {:5d} sequences'.format(len(trackers), len(dataset)))

    multiprocessing.set_start_method('spawn', force=True)

    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    if mode == 'sequential':
        for seq in dataset:
            for tracker_info in trackers:
                run_sequence(seq, tracker_info, debug=debug)
    elif mode == 'parallel':
        param_list = [(seq, tracker_info, debug, num_gpus) for seq, tracker_info in product(dataset, trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
    print('Done')
