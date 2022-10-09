# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import io
import itertools
import json
import logging
import numpy as np
import os
import tempfile
from collections import OrderedDict
from typing import Optional
from PIL import Image
from tabulate import tabulate
import torch
from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager

from detectron2.evaluation import DatasetEvaluator

logger = logging.getLogger(__name__)


class PSGEvaluator(DatasetEvaluator):
    """
    Evaluate Panoptic Quality metrics on COCO using PanopticAPI.
    It saves panoptic segmentation prediction in `output_dir`

    It contains a synchronize call and has to be called from all workers.
    """

    def __init__(self, dataset_name: str, output_dir: Optional[str] = None):
        """
        Args:
            dataset_name: name of the dataset
            output_dir: output directory to save results for evaluation.
        """
        self._metadata = MetadataCatalog.get(dataset_name)
        self._thing_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
        }
        self._stuff_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.stuff_dataset_id_to_contiguous_id.items()
        }

        self._output_dir = output_dir
        if self._output_dir is not None:
            PathManager.mkdirs(self._output_dir)

    def reset(self):
        self._predictions = []

    def _convert_category_id(self, segment_info):
        isthing = segment_info.pop("isthing", None)
        if isthing is None:
            # the model produces panoptic category id directly. No more conversion needed
            return segment_info
        if isthing is True:
            segment_info["category_id"] = self._thing_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        else:
            segment_info["category_id"] = self._stuff_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        return segment_info

    def process(self, inputs, outputs):
        from panopticapi.utils import id2rgb

        for input, output in zip(inputs, outputs):
            panoptic_img, segments_info = output["panoptic_seg"]
            panoptic_img = panoptic_img.cpu().numpy()
            relations = torch.topk(output["panoptic_relations"], 3)[1]
            relations = relations.cpu().detach().tolist()
            if segments_info is None:
                # If "segments_info" is None, we assume "panoptic_img" is a
                # H*W int32 image storing the panoptic_id in the format of
                # category_id * label_divisor + instance_id. We reserve -1 for
                # VOID label, and add 1 to panoptic_img since the official
                # evaluation script uses 0 for VOID label.
                label_divisor = self._metadata.label_divisor
                segments_info = []
                for panoptic_label in np.unique(panoptic_img):
                    if panoptic_label == -1:
                        # VOID region.
                        continue
                    pred_class = panoptic_label // label_divisor
                    isthing = (
                        pred_class
                        in self._metadata.thing_dataset_id_to_contiguous_id.values()
                    )
                    segments_info.append(
                        {
                            "id": int(panoptic_label) + 1,
                            "category_id": int(pred_class),
                            "isthing": bool(isthing),
                        }
                    )
                # Official evaluation script uses 0 for VOID label.
                panoptic_img += 1

            file_name = os.path.basename(input["file_name"])
            file_name_png = os.path.splitext(file_name)[0] + ".png"
            with io.BytesIO() as out:
                Image.fromarray(id2rgb(panoptic_img)).save(out, format="PNG")
                segments_info = [self._convert_category_id(x) for x in segments_info]
                self._predictions.append(
                    {
                        "image_id": input["image_id"],
                        "file_name": file_name_png,
                        "png_string": out.getvalue(),
                        "segments_info": segments_info,
                        "relations": relations,
                    }
                )

    def evaluate(self):
        comm.synchronize()

        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        if not comm.is_main_process():
            return

        # PanopticApi requires local files
        gt_json = PathManager.get_local_path(self._metadata.panoptic_json)
        gt_folder = PathManager.get_local_path(self._metadata.panoptic_root)

        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
            logger.info("Writing all panoptic predictions to {} ...".format(pred_dir))
            for p in self._predictions:
                with open(os.path.join(pred_dir, p["file_name"]), "wb") as f:
                    f.write(p.pop("png_string"))

            with open(gt_json, "r") as f:
                json_data = json.load(f)
            json_data["annotations"] = self._predictions

            output_dir = self._output_dir or pred_dir
            predictions_json = os.path.join(output_dir, "predictions.json")
            with PathManager.open(predictions_json, "w") as f:
                f.write(json.dumps(json_data))

            from panopticapi.evaluation import pq_compute

            with contextlib.redirect_stdout(io.StringIO()):
                pq_res = pq_compute(
                    gt_json,
                    PathManager.get_local_path(predictions_json),
                    gt_folder=gt_folder,
                    pred_folder=pred_dir,
                )

        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        res["PQ_th"] = 100 * pq_res["Things"]["pq"]
        res["SQ_th"] = 100 * pq_res["Things"]["sq"]
        res["RQ_th"] = 100 * pq_res["Things"]["rq"]
        res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]

        mean_recall = compute_recall(
            gt_json, PathManager.get_local_path(predictions_json)
        )
        results = OrderedDict(
            {"panoptic_seg": res, "relation_mean_recall": mean_recall}
        )

        _print_panoptic_results(pq_res)

        return results


def compute_recall(gt_json_file, pred_json_file):
    with open(gt_json_file, "r") as f:
        gt_json = json.load(f)
    with open(pred_json_file, "r") as f:
        pred_json = json.load(f)

    print("Evaluation prediction recall:")
    print("Ground truth:")
    print("\tJSON file: {}".format(gt_json_file))
    print("Prediction:")
    print("\tJSON file: {}".format(pred_json_file))

    pred_relations = {
        el["image_id"]: el["relations"] for el in pred_json["annotations"]
    }
    relations_matched_list = []
    for gt_ann in gt_json["annotations"]:
        image_id = gt_ann["image_id"]
        if image_id not in pred_relations:
            raise Exception("no prediction for the image with id: {}".format(image_id))
        gt_relation = [
            int(relation[-1])
            for relation in gt_ann["relations"]
            if int(relation[-1]) > 5
        ]
        relations_matched_list.append((gt_relation, pred_relations[image_id]))

    score_list = np.zeros([56, 2], dtype=int)
    for gt, pred in relations_matched_list:
        for gt_id in gt:
            # pos 0 for counting all existing relations
            score_list[gt_id][0] += 1
            if gt_id in pred:
                # pos 1 for counting relations that is recalled
                score_list[gt_id][1] += 1

    score_list = score_list[6:]
    # to avoid nan
    score_list[:, 0][score_list[:, 0] == 0] = 1
    meanrecall = np.mean(score_list[:, 1] / score_list[:, 0])

    metrics = {}
    metrics["mean_recall"] = meanrecall

    logger.info(f"Panoptic Relations Evaluation Results: {100* meanrecall}")
    return metrics


def _print_panoptic_results(pq_res):
    headers = ["", "PQ", "SQ", "RQ", "#categories"]
    data = []
    for name in ["All", "Things", "Stuff"]:
        row = (
            [name]
            + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]]
            + [pq_res[name]["n"]]
        )
        data.append(row)
    table = tabulate(
        data,
        headers=headers,
        tablefmt="pipe",
        floatfmt=".3f",
        stralign="center",
        numalign="center",
    )
    logger.info("Panoptic Evaluation Results:\n" + table)


if __name__ == "__main__":
    from detectron2.utils.logger import setup_logger

    logger = setup_logger()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-json")
    parser.add_argument("--gt-dir")
    parser.add_argument("--pred-json")
    parser.add_argument("--pred-dir")
    args = parser.parse_args()

    from panopticapi.evaluation import pq_compute

    with contextlib.redirect_stdout(io.StringIO()):
        pq_res = pq_compute(
            args.gt_json,
            args.pred_json,
            gt_folder=args.gt_dir,
            pred_folder=args.pred_dir,
        )
        _print_panoptic_results(pq_res)
