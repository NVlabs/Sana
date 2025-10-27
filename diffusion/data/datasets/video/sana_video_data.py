# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import os.path as osp
import random
import traceback
from functools import lru_cache
from glob import glob
from zipfile import ZipFile

import imageio.v3 as iio
import numpy as np
import torch
import torchvision.io as io
from termcolor import colored
from torch.utils.data import Dataset
from torchvision import transforms as T

from diffusion.data.builder import DATASETS
from diffusion.data.datasets.utils import *
from diffusion.data.transforms import ResizeCrop, ToTensorVideo, get_closest_ratio
from diffusion.data.wids import lru_json_load
from diffusion.utils.logger import get_root_logger


@DATASETS.register_module()
class SanaZipDataset(Dataset):
    def __init__(
        self,
        data_dir={},
        transform=None,
        load_vae_feat=False,
        load_text_feat=False,
        config=None,
        caption_proportion=None,
        json_cache_dir=None,
        vae_cache_dir: str = None,
        sort_dataset: bool = False,
        external_caption_suffixes: list = None,
        external_data_filter: dict = None,
        motion_score_file_thres: dict = None,
        motion_score_cal_type: str = "average",
        num_frames: int = None,
        target_fps: int = 16,
        resample_fps: bool = True,
        shuffle_dataset: bool = False,
        **kwargs,
    ):
        if external_caption_suffixes is None:
            external_caption_suffixes = []
        if external_data_filter is None:
            external_data_filter = {}
        if motion_score_file_thres is None:
            motion_score_file_thres = {}

        self.logger = (
            get_root_logger() if config is None else get_root_logger(osp.join(config.work_dir, "train_log.log"))
        )
        if load_vae_feat:
            assert vae_cache_dir is not None, "vae_cache_dir is required when load_vae_feat is True"
            print(colored(f"load_vae_feat: {load_vae_feat}, vae_cache_dir: {vae_cache_dir}", "yellow"))
        self.vae_cache_dir = vae_cache_dir
        self.transform = transform if not load_vae_feat else None
        self.load_vae_feat = load_vae_feat
        self.load_text_feat = load_text_feat
        self.caption_proportion = caption_proportion if caption_proportion is not None else {"prompt": 1.0}
        self.external_caption_suffixes = external_caption_suffixes
        self.default_prompt = "prompt"  # "Qwen2.5-VL"
        self.max_length = 300
        self.aspect_ratio = eval(kwargs.pop("aspect_ratio_type"))  # base aspect ratio

        data_dirs = data_dir if isinstance(data_dir, dict) else {"default": data_dir}
        self.dataset = []
        self.num_frames = num_frames
        self.target_fps = target_fps
        self.resample_fps = resample_fps
        self.failed_zip_files = set()
        self.failed_data = {}
        self.external_data_filter = external_data_filter
        self.motion_score_file_thres = motion_score_file_thres
        self.motion_score_cal_type = motion_score_cal_type
        self.shuffle_dataset = shuffle_dataset

        self.ratio_index = {}
        self.ratio_nums = {}
        for k, v in self.aspect_ratio.items():
            self.ratio_index[float(k)] = []
            self.ratio_nums[float(k)] = 0

        if json_cache_dir is None:
            json_cache_dir = "output/data_cache"
        self.json_cache_dir = json_cache_dir
        os.makedirs(json_cache_dir, exist_ok=True)

        self.dataset = []
        fileset = set()

        for name, data_path in data_dirs.items():
            data_path = osp.expanduser(data_path)

            zip_count = len(glob(f"{data_path}/*.zip"))
            dir_cache_name = self.generate_cache_filename(name, zip_count)
            dir_save_path = osp.join(json_cache_dir, dir_cache_name)

            if os.path.exists(dir_save_path):
                current_dict = json.load(open(dir_save_path))
                self.logger.info(f"Loaded cached dataset for {name} from {dir_save_path}, count: {len(current_dict)}")
            else:
                self.logger.warning(f"Cache file not found for {dir_save_path}, will generate cache file")
                base_cache_name = f"{name}-{zip_count}_cached_dataset.json"
                base_save_path = osp.join(json_cache_dir, base_cache_name)

                if os.path.exists(base_save_path):
                    current_dict = json.load(open(base_save_path))
                    self.logger.info(
                        f"Loaded base cached dataset for {name} from {base_save_path}, count: {len(current_dict)}"
                    )
                    self.logger.info(f"Will apply filters at runtime for {name}")
                else:
                    self.shuffle_dataset = True
                    self.logger.warning(colored(f"Caching base dataset for {name} to {base_save_path}", "red"))

                    current_dict = []

                    zip_files = glob(f"{data_path}/*.zip")
                    for zip_file in zip_files:
                        zip_file = os.path.abspath(zip_file)
                        try:
                            with ZipFile(zip_file, "r") as z:
                                external_captions = {}
                                for i in z.infolist():
                                    if i.filename.endswith(".json"):
                                        continue
                                    key, ext = osp.splitext(i.filename)
                                    if ext not in [".mp4", ".npy"]:
                                        continue
                                    json_name = f"{key}.json"
                                    hashkey = f"{name}@{key}"
                                    if hashkey in fileset:
                                        continue
                                    fileset.add(hashkey)
                                    unique_name, *_ = name.split("@")
                                    current_dict.append(
                                        {
                                            "info": {},
                                            "cache_key": f"{unique_name}/{key}",
                                            "key": key,
                                            "zip_file": zip_file,
                                            "ext": ext,
                                            "json_name": json_name,
                                            "dataset_name": name,
                                        }
                                    )
                        except Exception as e:
                            self.logger.warning(f"Skip corrupted zip file: {zip_file}, error: {str(e)}")
                            self.failed_zip_files.add(zip_file)
                            continue

                    if torch.distributed.get_rank() == 0:
                        json.dump(current_dict, open(base_save_path, "w"), indent=4)
                    self.logger.info(f"Saved base cache for {name}, video count: {len(current_dict)}")

            self.dataset.extend(current_dict)

            if torch.distributed.get_rank() == 0:
                self.logger.info(
                    colored(
                        f"name: {name}, video count: {len(current_dict)}, total video count: {len(self.dataset)}",
                        "green",
                    )
                )

        if sort_dataset:
            self.dataset.sort(key=lambda x: x["key"])
            self.logger.warning(colored("Sorted the dataset", "red"))
        elif self.shuffle_dataset:
            # shuffle by folder+zip combination
            zip_file_groups = {}
            for item in self.dataset:
                zip_file_path = item["zip_file"]
                parts = zip_file_path.split("/")
                if len(parts) >= 2:
                    folder_name = parts[-2]
                    zip_name = parts[-1]
                    group_key = f"{folder_name}/{zip_name}"
                else:
                    group_key = zip_file_path

                if group_key not in zip_file_groups:
                    zip_file_groups[group_key] = []
                zip_file_groups[group_key].append(item)

            group_keys_list = list(zip_file_groups.keys())
            random.shuffle(group_keys_list)

            # shuffle both group order and files within each folder/zip
            self.dataset = []
            for group_key in group_keys_list:
                group_items = zip_file_groups[group_key]
                random.shuffle(group_items)  # shuffle files within each zip
                self.dataset.extend(group_items)

            self.logger.warning(
                colored(
                    "Applied global shuffle by folder+zip combination (files within each folder/zip are also shuffled)",
                    "red",
                )
            )

        self.ori_imgs_nums = len(self)

        if len(self.external_caption_suffixes) > 0:
            self.logger.info(f"Loading external caption json from: original_filename{external_caption_suffixes}.json")
        if len(self.motion_score_file_thres) > 0:
            self.logger.info(f"Loading motion score json from: {self.motion_score_file_thres}")
            self.logger.info(f"Motion score cal type: {self.motion_score_cal_type}")

    @lru_cache(16)
    def open_zip_file(path: str):
        return ZipFile(path, "r")

    @lru_cache(maxsize=16)
    def lru_json_load(fpath):
        with open(fpath) as fp:
            return json.load(fp)

    def generate_cache_filename(self, dataset_name, dataset_count):
        if not self.external_data_filter or not self.num_frames or dataset_name not in self.external_data_filter:
            return f"{dataset_name}-{dataset_count}_cached_dataset.json"

        filter_parts = []
        for filter_name, filter_info in self.external_data_filter[dataset_name].items():
            clean_name = filter_name.lstrip("_")
            min_val = float(filter_info["min"])
            max_val = float(filter_info["max"])
            filter_parts.append(f"{clean_name}_{min_val}-{max_val}")

        if filter_parts:
            filter_str = "_".join(filter_parts)
            filename = f"{dataset_name}-{dataset_count}_{filter_str}_f{max(self.num_frames, 81)}_cached_dataset.json"
        else:
            filename = f"{dataset_name}-{dataset_count}_cached_dataset.json"

        return filename

    def weighted_sample_caption_type(self, info):
        """
        choose caption type according to caption_proportion, only choose available types.
        Guarantee: return a caption type that exists in info and is not None, or return None if none available.
        """
        available_caption_types = []
        available_weights = []

        for caption_type, weight in self.caption_proportion.items():
            if caption_type in info and info[caption_type] is not None:
                available_caption_types.append(caption_type)
                available_weights.append(weight)

        if not available_caption_types:
            # Prefer default prompt if it exists
            if self.default_prompt in info and info[self.default_prompt] is not None:
                return self.default_prompt
            # None indicates no usable caption is available
            return None

        selected_caption_type = random.choices(available_caption_types, weights=available_weights, k=1)[0]
        return selected_caption_type

    def getdata(self, idx):
        data = self.dataset[idx]
        self.key = data["key"]

        if data["zip_file"] in self.failed_zip_files:
            raise ValueError(f"Failed zip file: {data['zip_file']}")

        info = data["info"]
        cache_key = data["cache_key"]

        ext = data["ext"]
        z = SanaZipDataset.open_zip_file(data["zip_file"])
        with z.open(data["json_name"], "r") as f:
            info.update(json.load(f))

        # external caption file
        for suffix in self.external_caption_suffixes:
            caption_json_path = data["zip_file"].replace(".zip", f"{suffix}.json")
            if os.path.exists(caption_json_path):
                try:
                    caption_json = SanaZipDataset.lru_json_load(caption_json_path)
                except:
                    caption_json = {}
                if self.key in caption_json:
                    external_caption_info = caption_json[self.key]
                    if self.default_prompt in external_caption_info:
                        info.update({suffix.replace(".", "_"): external_caption_info[self.default_prompt]})
                    else:
                        info.update(external_caption_info[list(external_caption_info.keys())[0]])

        # data info
        data_info = {
            "cache_key": cache_key,
            "zip_file": data["zip_file"],
            "key": data["key"],
            "dataset_name": data["dataset_name"],
        }

        ori_h = info["height"] = float(info["height"])
        ori_w = info["width"] = float(info["width"])

        closest_size, closest_ratio = get_closest_ratio(ori_h, ori_w, self.aspect_ratio)
        closest_size = tuple(map(lambda x: int(x), closest_size))
        self.closest_ratio = closest_ratio

        data_info["img_hw"] = torch.tensor([ori_h, ori_w], dtype=torch.float32)
        data_info["aspect_ratio"] = closest_ratio

        fps = 16
        unimatch_ratio = 16 / fps

        # media data
        with z.open(data["key"] + ext, "r") as f:
            if ext in [".jpg", ".png", ".jpeg", ".webp"]:
                frame_data = iio.imread(f)
            elif ext == ".mp4":
                frame_data = iio.imread(f, plugin="pyav")
            elif ext == ".npy":
                frame_data = np.load(f)
                if "z" in frame_data:
                    frame_data = frame_data["z"]

        frame_data = frame_data[: self.num_frames]

        # motion score
        motion_suffix = ""
        for suffix, thres in self.motion_score_file_thres.items():
            if suffix != "_unimatch":
                continue
            data_filter_json_path = data["zip_file"].replace(".zip", f"{suffix}.json")
            if os.path.exists(data_filter_json_path):
                data_filter_json = SanaZipDataset.lru_json_load(data_filter_json_path)
                if self.key in data_filter_json:
                    data_filter_info = data_filter_json[self.key]
                    score_data = data_filter_info[next(iter(data_filter_info))]
                    if isinstance(score_data, int) or isinstance(score_data, float):
                        score = score_data
                    elif isinstance(score_data, list) and self.motion_score_cal_type == "average":
                        score = sum(score_data) / len(score_data)
                    elif isinstance(score_data, list) and self.motion_score_cal_type == "max":
                        score = max(score_data)
                    else:
                        raise ValueError(
                            f"Unknown score type: {type(score_data)}, {score_data} {self.motion_score_cal_type}"
                        )

                    motion_suffix = f" motion score: {int(score * unimatch_ratio)}."

        # caption selction
        caption_type = self.weighted_sample_caption_type(info)
        if caption_type is None:
            self.logger.warning(f"No available caption for data path: {data['zip_file']}")
            txt_fea = ""
            caption_type = "null"
        else:
            txt_fea = "" if info[caption_type] is None else info[caption_type]
        txt_fea = txt_fea + motion_suffix

        # transform
        if self.load_vae_feat:
            vframes = torch.from_numpy(frame_data).clone()  # C,F,H,W
        else:
            self.transform = T.Compose(
                [
                    ToTensorVideo(),  # TCHW
                    ResizeCrop(closest_size),
                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                ]
            )

            vframes = torch.from_numpy(frame_data).clone().permute(0, 3, 1, 2)
            vframes = self.transform(vframes)

        attention_mask = torch.ones(1, 1, self.max_length, dtype=torch.int16)  # 1x1xT

        # in case of ratio error
        if idx not in self.ratio_index[closest_ratio]:
            self.ratio_index[closest_ratio].append(idx)

        return (
            vframes,
            txt_fea,
            attention_mask.to(torch.int16),
            data_info,
            idx,
            caption_type,
            {"height": ori_h, "width": ori_w},
            0.0,
        )

    def __getitem__(self, idx):
        for _ in range(100):
            try:
                return self.getdata(idx)
            except Exception as e:
                traceback_str = traceback.format_exc()
                print(
                    f"__class__: {self.__class__.__name__}.getdata({idx}) Error details: {str(e)}, data path: {self.dataset[idx]['zip_file']}"
                    f"\n{traceback_str}"
                )
                idx = random.choice(self.ratio_index[self.closest_ratio])

        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.dataset)

    def get_data_info(self, idx):
        try:
            data = self.dataset[idx]
            info = data["info"]
            key = data["key"]
            ext = data["ext"]
            if "dataset_name" in data:
                dataset_name = data["dataset_name"]
            else:
                dataset_name = os.path.basename(os.path.dirname(data["zip_file"]))

            z = SanaZipDataset.open_zip_file(data["zip_file"])
            with z.open(data["json_name"], "r") as f:
                info.update(json.load(f))

            if "frames" in info:
                frame_num = int(info["frames"])
                if frame_num < self.num_frames or int(info["frames"]) < self.num_frames:
                    return None

            ori_h = info["height"] = float(info.get("height"))
            ori_w = info["width"] = float(info.get("width"))
            closest_size, closest_ratio = get_closest_ratio(ori_h, ori_w, self.aspect_ratio)

            return {
                "height": info["height"],
                "width": info["width"],
                "key": key,
                "index": idx,
                "zip_file": data["zip_file"],
                "ext": data["ext"],
                "closest_ratio": closest_ratio,
            }
        except Exception as e:
            traceback_str = traceback.format_exc()
            print(
                f"__class__: {self.__class__.__name__}.get_data_info() Error details: {str(e)}, data path: {data['zip_file']}"
                f"\n{traceback_str}"
            )
            return None


class DistributePromptsDataset(torch.utils.data.Dataset):
    """Dataset for other models inference.

    Args:
        prompts: Dictionary with keys and (prompt, image_path) tuples as values, or list of prompts
        original_indices: List of original indices from txt file corresponding to each prompt
    """

    def __init__(self, prompts, original_indices=None):
        if isinstance(prompts, dict):
            self.prompts = prompts
            self.keys_list = list(self.prompts.keys())
            self.original_indices = original_indices or list(range(len(prompts)))
        else:
            # Convert list to dict where key and value are the same
            self.prompts = {
                prompt[:50].split("/")[0] + str(hashlib.sha256(prompt.encode()).hexdigest())[:10]: prompt
                for prompt in prompts
            }
            self.keys_list = list(self.prompts.keys())
            self.original_indices = original_indices or list(range(len(prompts)))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        key = self.keys_list[idx]
        prompt = self.prompts[key]
        txt_line_idx = self.original_indices[idx]
        return {
            "key": key,
            "prompt": prompt,
            "global_idx": txt_line_idx,
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from diffusion.data.wids import DistributedRangedSampler
    from diffusion.utils.data_sampler import AspectRatioBatchSampler, AspectRatioBatchSamplerVideo

    image_size = (480, 832)  # 480x832
    batch_size = 4

    transform = T.Compose(
        [
            ToTensorVideo(),  # TCHW
            ResizeCrop(image_size),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    dataset = SanaZipDataset(
        data_dir={"video_toy_data": "data/video_toy_data"},
        transform=transform,
        load_vae_feat=False,
        aspect_ratio_type="ASPECT_RATIO_VIDEO_480_MS",
        num_frames=81,
    )
    sampler = DistributedRangedSampler(dataset, num_replicas=1, rank=0)
    batch_sampler = AspectRatioBatchSamplerVideo(
        sampler=sampler,
        dataset=dataset,
        batch_size=batch_size,
        aspect_ratios=dataset.aspect_ratio,
        drop_last=True,
        ratio_nums=dataset.ratio_nums,
        valid_num=0,
    )
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0)

    for data in tqdm(dataloader):
        # print(data[6])
        pass
