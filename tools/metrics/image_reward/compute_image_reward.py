import json
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import ImageReward as RM
from tqdm import tqdm

from tools.metrics.utils import tracker


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--json_path", type=str, default="./benchmark-prompts-dict.json")

    parser.add_argument("--img_path", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default="Sana")
    parser.add_argument("--txt_path", type=str, default=None)
    parser.add_argument("--sample_nums", type=int, default=100)
    parser.add_argument("--sample_per_prompt", default=10, type=int)

    # online logging setting
    parser.add_argument("--log_metric", type=str, default="metric")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--log_image_reward", action="store_true")
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


def main():
    txt_path = args.txt_path if args.txt_path is not None else args.img_path
    save_txt_path = os.path.join(txt_path, f"{args.exp_name}_sample{sample_nums}_image_reward.txt")
    if os.path.exists(save_txt_path):
        with open(save_txt_path) as f:
            image_reward_value = f.readlines()[0].strip()
        print(f"Image Reward {image_reward_value}: {args.exp_name}")
        return {args.exp_name: float(image_reward_value)}

    total_scores = 0
    count = 0
    for k, v in tqdm(
        prompt_json.items(), desc=f"ImageReward {args.sample_per_prompt} images / prompt: {args.exp_name}"
    ):
        for i in range(args.sample_per_prompt):
            img_path = os.path.join(args.img_path, args.exp_name, f"{k}_{i}.jpg")
            score = model.score(v["prompt"], img_path)
            total_scores += score
            count += 1

    image_reward_value = total_scores / count
    print(f"Image Reward {image_reward_value}: {args.exp_name}")
    with open(save_txt_path, "w") as file:
        file.write(str(image_reward_value))

    return {args.exp_name: image_reward_value}


if __name__ == "__main__":
    args = parse_args()
    sample_nums = args.sample_nums

    model = RM.load("ImageReward-v1.0")
    prompt_json = json.load(open(args.json_path))
    print(args.img_path, args.exp_name)
    args.exp_name = os.path.basename(args.exp_name) or os.path.dirname(args.exp_name)

    image_reward_result = main()

    if args.log_image_reward:
        tracker(args, image_reward_result, args.suffix_label, pattern=args.tracker_pattern, metric="ImageReward")
