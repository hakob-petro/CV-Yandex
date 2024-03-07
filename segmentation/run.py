#!/usr/bin/env python3

from json import dumps, load
from os import environ
from os.path import join
from sys import argv


def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            filename, class_id = line.rstrip("\n").split(",")
            res[filename] = float(class_id)
    return res


def save_csv(ious, filename):
    with open(filename, "w") as fhandle:
        print("filename,ious", file=fhandle)
        for filename in sorted(ious.keys()):
            # save last click ious
            print("%s,%f" % (filename, ious[filename][-1]), file=fhandle)


def check_test(data_dir):
    output_dir = join(data_dir, "output")
    output = read_csv(join(output_dir, "output.csv"))

    import numpy as np

    iou = np.mean(list(output.values()))

    res = "Ok, IoU %.4f" % iou
    if environ.get("CHECKER"):
        print(res)
    return res


def grade(data_path):
    results = load(open(join(data_path, "results.json")))
    result = results[-1]["status"]

    if not result.startswith("Ok"):
        res = {"description": "", "mark": 0}
    else:
        iou_str = result.split("IoU")[-1]
        iou = float(iou_str)
        if iou >= 0.81:
            mark = 10
        elif iou >= 0.80:
            mark = 9
        elif iou >= 0.79:
            mark = 8
        elif iou >= 0.78:
            mark = 7
        elif iou >= 0.77:
            mark = 6
        elif iou >= 0.76:
            mark = 5
        elif iou >= 0.75:
            mark = 4
        elif iou >= 0.60:
            mark = 3
        elif iou >= 0.30:
            mark = 2
        elif iou > 0:
            mark = 1
        else:
            mark = 0

        res = {"description": iou_str, "mark": mark}
    if environ.get("CHECKER"):
        print(dumps(res))
    return res


def run_single_test(data_dir, output_dir):
    import os
    import random
    from os.path import abspath, dirname
    from pathlib import Path

    import cv2
    import numpy as np
    import torch
    from solution import ISModel, Predictor
    from tqdm import tqdm
    from utils.clicker import Clicker
    from utils.datasets import TestDataset
    from utils.misc import draw_probmap, draw_with_blend_and_clicks

    def setup_deterministic(seed):
        torch.set_printoptions(precision=16)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.set_deterministic_debug_mode(2)

    def evaluate_dataset(dataset, predictor, **kwargs):
        setup_deterministic(42)
        all_ious = {}
        for index in tqdm(range(len(dataset)), leave=False):
            sample = dataset.get_sample(index)
            sample_ious = evaluate_sample(
                sample.image,
                sample.gt_mask,
                predictor,
                sample_id=index,
                **kwargs,
            )
            all_ious[sample.imname] = sample_ious
        return all_ious

    def evaluate_sample(
        image,
        gt_mask,
        predictor,
        pred_thr,
        max_clicks,
        sample_id=None,
        callback=None,
    ):
        clicker = Clicker(gt_mask=gt_mask)
        pred_mask = np.zeros_like(gt_mask)
        ious_list = []
        with torch.no_grad():
            predictor.set_input_image(image)
            for click_indx in range(max_clicks):
                clicker.make_next_click(pred_mask)
                pred_probs = predictor.get_prediction(clicker)
                pred_mask = pred_probs > pred_thr
                if callback is not None:
                    callback(
                        image,
                        pred_probs,
                        sample_id,
                        click_indx,
                        clicker.clicks_list,
                    )
                iou = get_iou(gt_mask, pred_mask)
                ious_list.append(iou)
            return np.array(ious_list, dtype=np.float32)

    def get_iou(gt_mask, pred_mask):
        obj_gt_mask = gt_mask == 1
        intersection = np.logical_and(pred_mask, obj_gt_mask).sum()
        union = np.logical_or(pred_mask, obj_gt_mask).sum()
        return intersection / union

    def get_prediction_vis_callback(save_path, prob_thresh):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        def callback(
            image,
            pred_probs,
            sample_id,
            click_indx,
            clicks_list,
        ):
            sample_path = save_path / f"{sample_id}_{click_indx}.jpg"
            prob_map = draw_probmap(pred_probs)
            image_with_mask = draw_with_blend_and_clicks(
                image,
                pred_probs > prob_thresh,
                clicks_list=clicks_list,
            )
            cv2.imwrite(
                str(sample_path),
                np.concatenate((image_with_mask, prob_map), axis=1)[:, :, ::-1],
            )

        return callback

    device = torch.device("cpu")
    dataset = TestDataset(images=data_dir + "/imgs", masks=data_dir + "/masks")
    model = ISModel()
    code_dir = dirname(abspath(__file__))
    model_path = join(code_dir, "checkpoint.pth")
    model.restore_from_checkpoint(model_path, device)
    model = model.eval()
    predictor = Predictor(model, device)
    pred_thr = model.pred_thr if hasattr(model, "pred_thr") else 0.5

    vis_callback = get_prediction_vis_callback(output_dir, pred_thr)
    all_ious = evaluate_dataset(
        dataset,
        predictor,
        pred_thr=pred_thr,
        max_clicks=3,
        callback=vis_callback,
    )
    save_csv(all_ious, join(output_dir, "output.csv"))


if __name__ == "__main__":
    if environ.get("CHECKER"):
        # Script is running in testing system
        if len(argv) != 4:
            print("Usage: %s mode data_dir output_dir" % argv[0])
            exit(0)

        mode = argv[1]
        data_dir = argv[2]
        output_dir = argv[3]

        if mode == "run_single_test":
            run_single_test(data_dir, output_dir)
        elif mode == "check_test":
            check_test(data_dir)
        elif mode == "grade":
            grade(data_dir)
    else:
        # Script is running locally, run on dir with tests
        if len(argv) != 2:
            print("Usage: %s tests_dir" % argv[0])
            exit(0)

        from glob import glob
        from json import dump
        from os import makedirs
        from os.path import basename
        from re import sub
        from time import time
        from traceback import format_exc

        tests_dir = argv[1]

        results = []
        for input_dir in sorted(glob(join(tests_dir, "[0-9][0-9]_*_input"))):
            output_dir = sub("input$", "check", input_dir)
            run_output_dir = join(output_dir, "output")
            makedirs(run_output_dir, exist_ok=True)
            traceback = None

            try:
                start = time()
                run_single_test(input_dir, run_output_dir)
                end = time()
                running_time = end - start
            except Exception:
                status = "Runtime error"
                traceback = format_exc()
            else:
                try:
                    status = check_test(output_dir)
                except Exception:
                    status = "Checker error"
                    traceback = format_exc()

            test_num = basename(input_dir)[:2]
            if status in ("Runtime error", "Checker error"):
                print(test_num, status, "\n", traceback)
                results.append({"status": status})
            else:
                print(test_num, "%.2fs" % running_time, status)
                results.append({"time": running_time, "status": status})

        dump(results, open(join(tests_dir, "results.json"), "w"))
        res = grade(tests_dir)
        print("Mark:", res["mark"], res["description"])
