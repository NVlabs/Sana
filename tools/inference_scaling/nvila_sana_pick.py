import json
import os
import os.path as osp
import shutil

import fire
import numpy as np
import PIL.Image
from tqdm import tqdm
from transformers import AutoModel

model = None


def load_model(model_name):
    global model, yes_id, no_id
    print("loading NVILA model")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
    yes_id = model.tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = model.tokenizer.encode("no", add_special_tokens=False)[0]
    print("loading NVILA finished")


def nvila_compare(
    prompt='a cyberpunk cat with a neon sign that says "Sana"',
    files=[f"output/sana_test_prompt/0.png", f"output/sana_test_prompt/1.png"],
):

    prompt = f"""You are an AI assistant specializing in image analysis and ranking. Your task is to analyze and compare image based on how well they match the given prompt.
The given prompt is:{prompt}. Please consider the prompt and the image to make a decision and response directly with 'yes' or 'no'.
"""

    r1, scores1 = model.generate_content([PIL.Image.open(files[0]), prompt])

    r2, scores2 = model.generate_content([PIL.Image.open(files[1]), prompt])

    if r1 == r2:
        if r1 == "yes":
            # pick the one with higher score for yes
            if scores1[0][0, yes_id] > scores2[0][0, yes_id]:
                return files[0]
            else:
                return files[1]
        else:
            # pick the one with less score for no
            if scores1[0][0, no_id] < scores2[0][0, no_id]:
                return files[0]
            else:
                return files[1]
    else:
        if r1 == "yes":
            return files[0]
        else:
            return files[1]


def get_prompt(idx, base_dir):
    # output/4800m_2048_v2/00000/metadata.jsonl
    with open(f"{base_dir}/{idx:05d}/metadata.jsonl") as f:
        for line in f:
            jinfo = json.loads(line)
            return jinfo["prompt"]
    return None


def get_files(idx, base_dir):
    output_dir = f"{base_dir}/{idx:05d}/samples"
    print(output_dir)
    files = []
    for file in os.listdir(output_dir):
        if file.endswith(".png"):
            files.append(os.path.join(output_dir, file))
    return files


def main(
    start_idx,
    end_idx,
    base_dir,
    model_name="Efficient-Large-Model/NVILA-Lite-2B-Verifier",
    number_of_files=2048,
    output_dir="output/nvila_pick",
    pick_number=4,
):
    # if pick_number and number_of_files are not 2^n, raise warning
    if pick_number != 2 ** int(np.log2(pick_number)) or number_of_files != 2 ** int(np.log2(number_of_files)):
        print(
            f"warning: pick_number and number_of_files are not 2^n, pick_number: {pick_number}, number_of_files: {number_of_files}"
        )
        pick_number = 2 ** int(np.log2(pick_number))
        number_of_files = 2 ** int(np.log2(number_of_files))
        print(f"warning: adjusted to 2^n, pick_number: {pick_number}, number_of_files: {number_of_files}")

    load_model(model_name)
    output_dir = f"{output_dir}/best_{pick_number}_of_{number_of_files}"

    for idx in range(start_idx, end_idx):
        files = get_files(idx, base_dir)
        files = files[:number_of_files]
        prompt = get_prompt(idx, base_dir)

        result_dir = osp.join(output_dir, base_dir, f"{idx:05d}")

        os.makedirs(result_dir, exist_ok=True)
        metadata_path = osp.join(base_dir, f"{idx:05d}", "metadata.jsonl")
        new_metadata_path = osp.join(output_dir, base_dir, f"{idx:05d}", "metadata.jsonl")
        shutil.copy(metadata_path, new_metadata_path)

        # if osp.join(output_dir, base_dir,  f"{idx:05d}", "samples") exists 4 files, skip
        print(f"checking {output_dir}/{base_dir}/{idx:05d}/samples files number")
        if (
            osp.exists(osp.join(output_dir, base_dir, f"{idx:05d}", "samples"))
            and len(os.listdir(osp.join(output_dir, base_dir, f"{idx:05d}", "samples"))) == 4
        ):
            print(f"skip {idx} because {base_dir}/{idx:05d}/samples exists")
            continue

        print(f"prompt: {prompt}, {len(files)} files")
        round = 0
        while len(files) > pick_number:
            _files = []
            bar = tqdm(range(0, len(files), 2), desc=f"Round {round}", leave=False)
            for idx in bar:
                choice = nvila_compare(prompt, [files[idx], files[idx + 1]])
                _files.append(choice)
                bar.set_description_str(f"Round {round}, evaluating {idx}, {choice}")
            files = _files
            round += 1
            print(f"Round {round}, {len(files)} files left")

        print(files)
        for f in files:
            fpath = osp.join(output_dir, f)
            os.makedirs(osp.dirname(fpath), exist_ok=True)
            shutil.copy(f, fpath)


if __name__ == "__main__":
    fire.Fire(main)
