import argparse
import os
import os.path as osp
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")  # ignore warning
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent.parent.parent))

from tools.metrics.utils import tracker


def parse_args():
    parser = argparse.ArgumentParser(description="DPG-Bench evaluation.")
    parser.add_argument("--image-root-path", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default="Sana")
    parser.add_argument("--txt_path", type=str, default=None)
    parser.add_argument("--sample_nums", type=int, default=1065)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--csv", type=str, default="tools/metrics/dpg_bench/dpg_bench.csv")
    parser.add_argument("--res-path", type=str, default=None)
    parser.add_argument("--pic-num", type=int, default=1)
    parser.add_argument("--vqa-model", type=str, default="mplug")

    # online logging setting
    parser.add_argument("--log_metric", type=str, default="metric")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--log_dpg", action="store_true")
    parser.add_argument("--suffix_label", type=str, default="", help="used for image-reward online log")
    parser.add_argument("--tracker_pattern", type=str, default="epoch_step", help="used for image-reward online log")
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
    return args


class MPLUG(torch.nn.Module):
    def __init__(self, ckpt="damo/mplug_visual-question-answering_coco_large_en", device="gpu"):
        super().__init__()
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.pipeline_vqa = pipeline(Tasks.visual_question_answering, model=ckpt, device=device)

    def vqa(self, image, question):
        input_vqa = {"image": image, "question": question}
        result = self.pipeline_vqa(input_vqa)
        return result["text"]


def prepare_dpg_data(args):
    previous_id = ""
    current_id = ""
    question_dict = dict()
    category_count = defaultdict(int)
    # 'item_id', 'text', 'keywords', 'proposition_id', 'dependency', 'category_broad', 'category_detailed', 'tuple', 'question_natural_language'
    data = pd.read_csv(args.csv)
    for i, line in data.iterrows():
        if i == 0:
            continue

        current_id = line.item_id
        qid = int(line.proposition_id)
        dependency_list_str = line.dependency.split(",")
        dependency_list_int = []
        for d in dependency_list_str:
            d_int = int(d.strip())
            dependency_list_int.append(d_int)

        if current_id == previous_id:
            question_dict[current_id]["qid2tuple"][qid] = line.tuple
            question_dict[current_id]["qid2dependency"][qid] = dependency_list_int
            question_dict[current_id]["qid2question"][qid] = line.question_natural_language
        else:
            question_dict[current_id] = dict(
                qid2tuple={qid: line.tuple},
                qid2dependency={qid: dependency_list_int},
                qid2question={qid: line.question_natural_language},
            )

        category = line.question_natural_language.split("(")[0].strip()
        category_count[category] += 1

        previous_id = current_id

    return question_dict


def crop_image(input_image, crop_tuple=None):
    if crop_tuple is None:
        return input_image

    cropped_image = input_image.crop((crop_tuple[0], crop_tuple[1], crop_tuple[2], crop_tuple[3]))

    return cropped_image


def compute_dpg_one_sample(args, question_dict, image_path, vqa_model, resolution):
    generated_image = Image.open(image_path)
    crop_tuples_list = [
        (0, 0, resolution, resolution),
        (resolution, 0, resolution * 2, resolution),
        (0, resolution, resolution, resolution * 2),
        (resolution, resolution, resolution * 2, resolution * 2),
    ]

    crop_tuples = crop_tuples_list[: args.pic_num]
    key = osp.basename(image_path).split(".")[0]
    value = question_dict.get(key, None)
    qid2tuple = value["qid2tuple"]
    qid2question = value["qid2question"]
    qid2dependency = value["qid2dependency"]

    qid2answer = dict()
    qid2scores = dict()
    qid2validity = dict()

    scores = []
    for crop_tuple in crop_tuples:
        cropped_image = crop_image(generated_image, crop_tuple)
        for id, question in qid2question.items():
            answer = vqa_model.vqa(cropped_image, question)
            qid2answer[id] = answer
            qid2scores[id] = float(answer == "yes")
            with open(args.res_path.replace(".txt", "_detail.txt"), "a") as f:
                f.write(image_path + ", " + str(crop_tuple) + ", " + question + ", " + answer + "\n")
        qid2scores_orig = qid2scores.copy()

        for id, parent_ids in qid2dependency.items():
            # zero-out scores if parent questions are answered 'no'
            any_parent_answered_no = False
            for parent_id in parent_ids:
                if parent_id == 0:
                    continue
                if qid2scores[parent_id] == 0:
                    any_parent_answered_no = True
                    break
            if any_parent_answered_no:
                qid2scores[id] = 0
                qid2validity[id] = False
            else:
                qid2validity[id] = True

        score = sum(qid2scores.values()) / len(qid2scores)
        scores.append(score)
    average_score = sum(scores) / len(scores)
    with open(args.res_path, "a") as f:
        f.write(image_path + ", " + ", ".join(str(i) for i in scores) + ", " + str(average_score) + "\n")

    return average_score, qid2tuple, qid2scores_orig


def main():

    accelerator = Accelerator()

    question_dict = prepare_dpg_data(args)

    txt_path = args.txt_path if args.txt_path is not None else args.image_root_path
    args.image_root_path = osp.join(args.image_root_path, args.exp_name)
    sample_nums = args.sample_nums
    args.res_path = osp.join(txt_path, f"{args.exp_name}_sample{sample_nums}_dpg_results.txt")
    save_txt_path = osp.join(txt_path, f"{args.exp_name}_sample{sample_nums}_dpg_results_simple.txt")
    if os.path.exists(save_txt_path):
        with open(save_txt_path) as f:
            dpg_value = f.readlines()[0].strip()
        print(f"DPG-Bench: {dpg_value}: {args.exp_name}")
        return {args.exp_name: float(dpg_value)}

    if accelerator.is_main_process:
        with open(args.res_path, "w") as f:
            pass
        with open(args.res_path.replace(".txt", "_detail.txt"), "w") as f:
            pass

    device = str(accelerator.device)
    if args.vqa_model == "mplug":
        vqa_model = MPLUG(device=device)
    else:
        raise NotImplementedError
    vqa_model = accelerator.prepare(vqa_model)
    vqa_model = getattr(vqa_model, "module", vqa_model)

    filename_list = os.listdir(args.image_root_path)
    num_each_rank = len(filename_list) / accelerator.num_processes
    local_rank = accelerator.process_index
    local_filename_list = filename_list[round(local_rank * num_each_rank) : round((local_rank + 1) * num_each_rank)]

    local_scores = []
    local_category2scores = defaultdict(list)
    model_id = osp.basename(args.image_root_path)
    print(f"Start to conduct evaluation of {model_id}")
    for fn in tqdm(local_filename_list):
        image_path = osp.join(args.image_root_path, fn)
        try:
            # compute score of one sample
            score, qid2tuple, qid2scores = compute_dpg_one_sample(
                args=args,
                question_dict=question_dict,
                image_path=image_path,
                vqa_model=vqa_model,
                resolution=args.resolution,
            )
            local_scores.append(score)

            # summarize scores by categoris
            for qid in qid2tuple.keys():
                category = qid2tuple[qid].split("(")[0].strip()
                qid_score = qid2scores[qid]
                local_category2scores[category].append(qid_score)

        except Exception as e:
            print("Failed filename:", fn, e)
            continue

    accelerator.wait_for_everyone()
    global_dpg_scores = gather_object(local_scores)
    mean_dpg_score = np.mean(global_dpg_scores)

    global_categories = gather_object(list(local_category2scores.keys()))
    global_categories = set(global_categories)
    global_category2scores = dict()
    global_average_scores = []
    for category in global_categories:
        local_category_scores = local_category2scores.get(category, [])
        global_category2scores[category] = gather_object(local_category_scores)
        global_average_scores.extend(gather_object(local_category_scores))

    global_category2scores_l1 = defaultdict(list)
    for category in global_categories:
        l1_category = category.split("-")[0].strip()
        global_category2scores_l1[l1_category].extend(global_category2scores[category])

    time.sleep(3)
    if accelerator.is_main_process:
        output = f"Model: {model_id}\n"

        output += "L1 category scores:\n"
        for l1_category in global_category2scores_l1.keys():
            output += f"\t{l1_category}: {np.mean(global_category2scores_l1[l1_category]) * 100}\n"

        output += "L2 category scores:\n"
        for category in sorted(global_categories):
            output += f"\t{category}: {np.mean(global_category2scores[category]) * 100}\n"

        output += f"Image path: {args.image_root_path}\n"
        output += f"Save results to: {args.res_path}\n"
        output += f"DPG-Bench score: {mean_dpg_score * 100}"
        with open(args.res_path, "a") as f:
            f.write(output + "\n")
        print(output)

    if accelerator.is_main_process:
        with open(save_txt_path, "w") as file:
            file.write(str(mean_dpg_score * 100))

    return {args.exp_name: mean_dpg_score * 100}


if __name__ == "__main__":
    args = parse_args()
    args.exp_name = os.path.basename(args.exp_name) or os.path.dirname(args.exp_name)

    dpg_result = main()

    if args.log_dpg:
        tracker(args, dpg_result, args.suffix_label, pattern=args.tracker_pattern, metric="DPG")
