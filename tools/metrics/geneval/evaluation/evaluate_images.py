"""
Evaluate generated images using Mask2Former (or other object detector model)
"""

import argparse
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent.parent.parent.parent))
warnings.filterwarnings("ignore")

import mmdet
import numpy as np
import open_clip
import pandas as pd
import torch
from clip_benchmark.metrics import zeroshot_classification as zsc
from mmdet.apis import inference_detector, init_detector
from PIL import Image, ImageOps
from tqdm import tqdm

zsc.tqdm = lambda it, *args, **kwargs: it
from tools.metrics.utils import tracker

# Get directory path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda"


def timed(fn):
    def wrapper(*args, **kwargs):
        startt = time.time()
        result = fn(*args, **kwargs)
        endt = time.time()
        print(f"Function {fn.__name__!r} executed in {endt - startt:.3f}s", file=sys.stderr)
        return result

    return wrapper


# Load models
@timed
def load_models(args):
    CONFIG_PATH = args.model_config
    OBJECT_DETECTOR = args.options.get("model", "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco")
    CKPT_PATH = os.path.join(args.model_path, f"{OBJECT_DETECTOR}.pth")
    object_detector = init_detector(CONFIG_PATH, CKPT_PATH, device=DEVICE)

    clip_arch = args.options.get("clip_model", "ViT-L-14")
    clip_model, _, transform = open_clip.create_model_and_transforms(clip_arch, pretrained="openai", device=DEVICE)
    tokenizer = open_clip.get_tokenizer(clip_arch)

    with open(os.path.join(os.path.dirname(__file__), "object_names.txt")) as cls_file:
        classnames = [line.strip() for line in cls_file]

    return object_detector, (clip_model, transform, tokenizer), classnames


COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
COLOR_CLASSIFIERS = {}


# Evaluation parts
class ImageCrops(torch.utils.data.Dataset):
    def __init__(self, image: Image.Image, objects):
        self._image = image.convert("RGB")
        bgcolor = args.options.get("bgcolor", "#999")
        if bgcolor == "original":
            self._blank = self._image.copy()
        else:
            self._blank = Image.new("RGB", image.size, color=bgcolor)
        self._objects = objects

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, index):
        box, mask = self._objects[index]
        if mask is not None:
            assert tuple(self._image.size[::-1]) == tuple(mask.shape), (index, self._image.size[::-1], mask.shape)
            image = Image.composite(self._image, self._blank, Image.fromarray(mask))
        else:
            image = self._image
        if args.options.get("crop", "1") == "1":
            image = image.crop(box[:4])
        # if args.save:
        #     base_count = len(os.listdir(args.save))
        #     image.save(os.path.join(args.save, f"cropped_{base_count:05}.png"))
        return (transform(image), 0)


def color_classification(image, bboxes, classname):
    if classname not in COLOR_CLASSIFIERS:
        COLOR_CLASSIFIERS[classname] = zsc.zero_shot_classifier(
            clip_model,
            tokenizer,
            COLORS,
            [
                f"a photo of a {{c}} {classname}",
                f"a photo of a {{c}}-colored {classname}",
                f"a photo of a {{c}} object",
            ],
            DEVICE,
        )
    clf = COLOR_CLASSIFIERS[classname]
    dataloader = torch.utils.data.DataLoader(ImageCrops(image, bboxes), batch_size=16, num_workers=4)
    with torch.no_grad():
        pred, _ = zsc.run_classification(clip_model, clf, dataloader, DEVICE)
        return [COLORS[index.item()] for index in pred.argmax(1)]


def compute_iou(box_a, box_b):
    area_fn = lambda box: max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
    i_area = area_fn(
        [max(box_a[0], box_b[0]), max(box_a[1], box_b[1]), min(box_a[2], box_b[2]), min(box_a[3], box_b[3])]
    )
    u_area = area_fn(box_a) + area_fn(box_b) - i_area
    return i_area / u_area if u_area else 0


def relative_position(obj_a, obj_b):
    """Give position of A relative to B, factoring in object dimensions"""
    boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    #
    revised_offset = np.maximum(np.abs(offset) - POSITION_THRESHOLD * (dim_a + dim_b), 0) * np.sign(offset)
    if np.all(np.abs(revised_offset) < 1e-3):
        return set()
    #
    dx, dy = revised_offset / np.linalg.norm(offset)
    relations = set()
    if dx < -0.5:
        relations.add("left of")
    if dx > 0.5:
        relations.add("right of")
    if dy < -0.5:
        relations.add("above")
    if dy > 0.5:
        relations.add("below")
    return relations


def evaluate(image, objects, metadata):
    """
    Evaluate given image using detected objects on the global metadata specifications.
    Assumptions:
    * Metadata combines 'include' clauses with AND, and 'exclude' clauses with OR
    * All clauses are independent, i.e., duplicating a clause has no effect on the correctness
    * CHANGED: Color and position will only be evaluated on the most confidently predicted objects;
        therefore, objects are expected to appear in sorted order
    """
    correct = True
    reason = []
    matched_groups = []
    # Check for expected objects
    for req in metadata.get("include", []):
        classname = req["class"]
        matched = True
        found_objects = objects.get(classname, [])[: req["count"]]
        if len(found_objects) < req["count"]:
            correct = matched = False
            reason.append(f"expected {classname}>={req['count']}, found {len(found_objects)}")
        else:
            if "color" in req:
                # Color check
                colors = color_classification(image, found_objects, classname)
                if colors.count(req["color"]) < req["count"]:
                    correct = matched = False
                    reason.append(
                        f"expected {req['color']} {classname}>={req['count']}, found "
                        + f"{colors.count(req['color'])} {req['color']}; and "
                        + ", ".join(f"{colors.count(c)} {c}" for c in COLORS if c in colors)
                    )
            if "position" in req and matched:
                # Relative position check
                expected_rel, target_group = req["position"]
                if matched_groups[target_group] is None:
                    correct = matched = False
                    reason.append(f"no target for {classname} to be {expected_rel}")
                else:
                    for obj in found_objects:
                        for target_obj in matched_groups[target_group]:
                            true_rels = relative_position(obj, target_obj)
                            if expected_rel not in true_rels:
                                correct = matched = False
                                reason.append(
                                    f"expected {classname} {expected_rel} target, found "
                                    + f"{' and '.join(true_rels)} target"
                                )
                                break
                        if not matched:
                            break
        if matched:
            matched_groups.append(found_objects)
        else:
            matched_groups.append(None)
    # Check for non-expected objects
    for req in metadata.get("exclude", []):
        classname = req["class"]
        if len(objects.get(classname, [])) >= req["count"]:
            correct = False
            reason.append(f"expected {classname}<{req['count']}, found {len(objects[classname])}")
    return correct, "\n".join(reason)


def evaluate_image(filepath, metadata):
    result = inference_detector(object_detector, filepath)
    bbox = result[0] if isinstance(result, tuple) else result
    segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None
    image = ImageOps.exif_transpose(Image.open(filepath))
    detected = {}
    # Determine bounding boxes to keep
    confidence_threshold = THRESHOLD if metadata["tag"] != "counting" else COUNTING_THRESHOLD
    for index, classname in enumerate(classnames):
        ordering = np.argsort(bbox[index][:, 4])[::-1]
        ordering = ordering[bbox[index][ordering, 4] > confidence_threshold]  # Threshold
        ordering = ordering[:MAX_OBJECTS].tolist()  # Limit number of detected objects per class
        detected[classname] = []
        while ordering:
            max_obj = ordering.pop(0)
            detected[classname].append((bbox[index][max_obj], None if segm is None else segm[index][max_obj]))
            ordering = [
                obj
                for obj in ordering
                if NMS_THRESHOLD == 1 or compute_iou(bbox[index][max_obj], bbox[index][obj]) < NMS_THRESHOLD
            ]
        if not detected[classname]:
            del detected[classname]
    # Evaluate
    is_correct, reason = evaluate(image, detected, metadata)
    return {
        "filename": filepath,
        "tag": metadata["tag"],
        "prompt": metadata["prompt"],
        "correct": is_correct,
        "reason": reason,
        "metadata": json.dumps(metadata),
        "details": json.dumps({key: [box.tolist() for box, _ in value] for key, value in detected.items()}),
    }


def main(args):
    full_results = []
    image_dir = str(os.path.join(args.img_path, args.exp_name))
    args.outfile = f"{image_dir}_geneval.jsonl"

    if os.path.exists(args.outfile):
        df = pd.read_json(args.outfile, orient="records", lines=True)
        return {args.exp_name: df}

    for subfolder in tqdm(os.listdir(image_dir), f"Detecting on {args.gpu_id}"):
        folderpath = os.path.join(image_dir, subfolder)
        if not os.path.isdir(folderpath) or not subfolder.isdigit():
            continue
        with open(os.path.join(folderpath, "metadata.jsonl")) as fp:
            metadata = json.load(fp)
        # Evaluate each image
        for imagename in os.listdir(os.path.join(folderpath, "samples")):
            imagepath = os.path.join(folderpath, "samples", imagename)
            if not os.path.isfile(imagepath) or not re.match(r"\d+\.png", imagename):
                continue
            result = evaluate_image(imagepath, metadata)
            full_results.append(result)

    # Save results
    if os.path.dirname(args.outfile):
        os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    with open(args.outfile, "w") as fp:
        pd.DataFrame(full_results).to_json(fp, orient="records", lines=True)
    df = pd.read_json(args.outfile, orient="records", lines=True)

    return {args.exp_name: df}


def tracker_ori(df_dict, label=""):
    if args.report_to == "wandb":
        import wandb

        wandb_name = f"[{args.log_metric}]_[{args.name}]"
        wandb.init(project=args.tracker_project_name, name=wandb_name, resume="allow", id=wandb_name, tags="metrics")
        run = wandb.run
        run.define_metric("custom_step")
        run.define_metric(f"GenEval_Overall_Score({label})", step_metric="custom_step")

        for exp_name, df in df_dict.items():
            steps = []

            # Initialize the wandb table inside the function.
            wandb_table = wandb.Table(columns=["Metric", "Value"])

            # Compute total images, total prompts, correct image percentage, and correct prompt percentage.
            total_images = len(df)
            total_prompts = len(df.groupby("metadata"))
            percentage_correct_images = df["correct"].mean()
            percentage_correct_prompts = df.groupby("metadata")["correct"].any().mean()

            wandb_table.add_data("Total images", total_images)
            wandb_table.add_data("Total prompts", total_prompts)
            wandb_table.add_data("% correct images", f"{percentage_correct_images:.2%}")
            wandb_table.add_data("% correct prompts", f"{percentage_correct_prompts:.2%}")

            task_scores = []
            for tag, task_df in df.groupby("tag", sort=False):
                task_score = task_df["correct"].mean()
                task_scores.append(task_score)
                task_result = f"{tag:<16} = {task_score:.2%} ({task_df['correct'].sum()} / {len(task_df)})"
                print(task_result)

                # Add the task score to the table.
                wandb_table.add_data(tag, f"{task_score:.2%} ({task_df['correct'].sum()} / {len(task_df)})")

            # Compute the overall score.
            overall_score = np.mean(task_scores)
            print(f"Overall score (avg. over tasks): {overall_score:.5f}")

            # Extract the step from exp_name.
            match = re.search(r".*epoch(\d+)_step(\d+).*", exp_name)
            if match:
                epoch_name, step_name = match.groups()
                step = int(step_name)
                steps.append(step)

                # Log each step and its corresponding overall score.
                run.log({"custom_step": step, f"GenEval_Overall_Score({label})": overall_score})

            # Log the table to wandb.
            run.log({"Metrics Table": wandb_table})

    else:
        print(f"{args.report_to} is not supported")


def log_results(df_dict):
    # Measure overall success

    for exp_name, df in df_dict.items():
        print("Summary")
        print("=======")
        print(f"Total images: {len(df)}")
        print(f"Total prompts: {len(df.groupby('metadata'))}")
        print(f"% correct images: {df['correct'].mean():.2%}")
        print(f"% correct prompts: {df.groupby('metadata')['correct'].any().mean():.2%}")
        print()

        # By group

        task_scores = []

        print("Task breakdown")
        print("==============")
        for tag, task_df in df.groupby("tag", sort=False):
            task_scores.append(task_df["correct"].mean())
            print(f"{tag:<16} = {task_df['correct'].mean():.2%} ({task_df['correct'].sum()} / {len(task_df)})")
        print()

        print(f"Overall score (avg. over tasks): {np.mean(task_scores):.5f}")

        return {exp_name: np.mean(task_scores)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default="Sana")
    parser.add_argument("--outfile", type=str, default="results.jsonl")
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    # Other arguments
    parser.add_argument("--options", nargs="*", type=str, default=[])
    # wandb report
    parser.add_argument("--log_geneval", action="store_true")
    parser.add_argument("--log_metric", type=str, default="metric")
    parser.add_argument("--suffix_label", type=str, default="", help="used for clip_score online log")
    parser.add_argument("--tracker_pattern", type=str, default="epoch_step", help="used for GenEval online log")
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="t2i-evit-baseline",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        default="baseline",
        help=("Wandb Project Name"),
    )

    args = parser.parse_args()
    args.options = dict(opt.split("=", 1) for opt in args.options)
    if args.model_config is None:
        args.model_config = os.path.join(
            os.path.dirname(mmdet.__file__), "../configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
        )
    return args


if __name__ == "__main__":
    args = parse_args()
    object_detector, (clip_model, transform, tokenizer), classnames = load_models(args)
    THRESHOLD = float(args.options.get("threshold", 0.3))
    COUNTING_THRESHOLD = float(args.options.get("counting_threshold", 0.9))
    MAX_OBJECTS = int(args.options.get("max_objects", 16))
    NMS_THRESHOLD = float(args.options.get("max_overlap", 1.0))
    POSITION_THRESHOLD = float(args.options.get("position_threshold", 0.1))

    args.exp_name = os.path.basename(args.exp_name) or os.path.dirname(args.exp_name)
    df_dict = main(args)
    geneval_result = log_results(df_dict)
    if args.log_geneval:
        # tracker_ori(df_dict, args.suffix_label)
        tracker(args, geneval_result, args.suffix_label, pattern=args.tracker_pattern, metric="GenEval")
