"""
Generate prompts for evaluation
"""

import argparse
import json
import os

import numpy as np
import yaml

# Load classnames

with open("object_names.txt") as cls_file:
    classnames = [line.strip() for line in cls_file]

# Proper a vs an


def with_article(name: str):
    if name[0] in "aeiou":
        return f"an {name}"
    return f"a {name}"


# Proper plural


def make_plural(name: str):
    if name[-1] in "s":
        return f"{name}es"
    return f"{name}s"


# Generates single object samples


def generate_single_object_sample(rng: np.random.Generator, size: int = None):
    TAG = "single_object"
    if size > len(classnames):
        size = len(classnames)
        print(f"Not enough distinct classes, generating only {size} samples")
    return_scalar = size is None
    size = size or 1
    idxs = rng.choice(len(classnames), size=size, replace=False)
    samples = [
        dict(
            tag=TAG,
            include=[{"class": classnames[idx], "count": 1}],
            prompt=f"a photo of {with_article(classnames[idx])}",
        )
        for idx in idxs
    ]
    if return_scalar:
        return samples[0]
    return samples


# Generate two object samples


def generate_two_object_sample(rng: np.random.Generator):
    TAG = "two_object"
    idx_a, idx_b = rng.choice(len(classnames), size=2, replace=False)
    return dict(
        tag=TAG,
        include=[{"class": classnames[idx_a], "count": 1}, {"class": classnames[idx_b], "count": 1}],
        prompt=f"a photo of {with_article(classnames[idx_a])} and {with_article(classnames[idx_b])}",
    )


# Generate counting samples

numbers = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


def generate_counting_sample(rng: np.random.Generator, max_count=4):
    TAG = "counting"
    idx = rng.choice(len(classnames))
    num = int(rng.integers(2, max_count, endpoint=True))
    return dict(
        tag=TAG,
        include=[{"class": classnames[idx], "count": num}],
        exclude=[{"class": classnames[idx], "count": num + 1}],
        prompt=f"a photo of {numbers[num]} {make_plural(classnames[idx])}",
    )


# Generate color samples

colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]


def generate_color_sample(rng: np.random.Generator):
    TAG = "colors"
    idx = rng.choice(len(classnames) - 1) + 1
    idx = (idx + classnames.index("person")) % len(classnames)  # No "[COLOR] person" prompts
    color = colors[rng.choice(len(colors))]
    return dict(
        tag=TAG,
        include=[{"class": classnames[idx], "count": 1, "color": color}],
        prompt=f"a photo of {with_article(color)} {classnames[idx]}",
    )


# Generate position samples

positions = ["left of", "right of", "above", "below"]


def generate_position_sample(rng: np.random.Generator):
    TAG = "position"
    idx_a, idx_b = rng.choice(len(classnames), size=2, replace=False)
    position = positions[rng.choice(len(positions))]
    return dict(
        tag=TAG,
        include=[
            {"class": classnames[idx_b], "count": 1},
            {"class": classnames[idx_a], "count": 1, "position": (position, 0)},
        ],
        prompt=f"a photo of {with_article(classnames[idx_a])} {position} {with_article(classnames[idx_b])}",
    )


# Generate color attribution samples


def generate_color_attribution_sample(rng: np.random.Generator):
    TAG = "color_attr"
    idxs = rng.choice(len(classnames) - 1, size=2, replace=False) + 1
    idx_a, idx_b = (idxs + classnames.index("person")) % len(classnames)  # No "[COLOR] person" prompts
    cidx_a, cidx_b = rng.choice(len(colors), size=2, replace=False)
    return dict(
        tag=TAG,
        include=[
            {"class": classnames[idx_a], "count": 1, "color": colors[cidx_a]},
            {"class": classnames[idx_b], "count": 1, "color": colors[cidx_b]},
        ],
        prompt=f"a photo of {with_article(colors[cidx_a])} {classnames[idx_a]} and {with_article(colors[cidx_b])} {classnames[idx_b]}",
    )


# Generate evaluation suite


def generate_suite(rng: np.random.Generator, n: int = 100, output_path: str = ""):
    samples = []
    # Generate single object samples for all COCO classnames
    samples.extend(generate_single_object_sample(rng, size=len(classnames)))
    # Generate two object samples (~100)
    for _ in range(n):
        samples.append(generate_two_object_sample(rng))
    # Generate counting samples
    for _ in range(n):
        samples.append(generate_counting_sample(rng, max_count=4))
    # Generate color samples
    for _ in range(n):
        samples.append(generate_color_sample(rng))
    # Generate position samples
    for _ in range(n):
        samples.append(generate_position_sample(rng))
    # Generate color attribution samples
    for _ in range(n):
        samples.append(generate_color_attribution_sample(rng))
    # De-duplicate
    unique_samples, used_samples = [], set()
    for sample in samples:
        sample_text = yaml.safe_dump(sample)
        if sample_text not in used_samples:
            unique_samples.append(sample)
            used_samples.add(sample_text)

    # Write to files
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "generation_prompts.txt"), "w") as fp:
        for sample in unique_samples:
            print(sample["prompt"], file=fp)
    with open(os.path.join(output_path, "evaluation_metadata.jsonl"), "w") as fp:
        for sample in unique_samples:
            print(json.dumps(sample), file=fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=43, help="generation seed (default: 43)")
    parser.add_argument("--num-prompts", "-n", type=int, default=100, help="number of prompts per task (default: 100)")
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        default="prompts",
        help="output folder for prompts and metadata (default: 'prompts/')",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    generate_suite(rng, args.num_prompts, args.output_path)
