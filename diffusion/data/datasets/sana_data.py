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

# This file is modified from https://github.com/PixArt-alpha/PixArt-sigma
import getpass
import json
import os
import os.path as osp
import random

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from termcolor import colored
from torch.utils.data import Dataset

from diffusion.data.builder import DATASETS, get_data_path
from diffusion.data.wids import ShardListDataset, ShardListDatasetMulti, lru_json_load
from diffusion.utils.logger import get_root_logger


@DATASETS.register_module()
class SanaImgDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir="",
        transform=None,
        resolution=256,
        load_vae_feat=False,
        load_text_feat=False,
        max_length=300,
        config=None,
        caption_proportion=None,
        external_caption_suffixes=None,
        external_clipscore_suffixes=None,
        clip_thr=0.0,
        clip_thr_temperature=1.0,
        img_extension=".png",
        **kwargs,
    ):
        if external_caption_suffixes is None:
            external_caption_suffixes = []
        if external_clipscore_suffixes is None:
            external_clipscore_suffixes = []

        self.logger = (
            get_root_logger() if config is None else get_root_logger(osp.join(config.work_dir, "train_log.log"))
        )
        self.transform = transform if not load_vae_feat else None
        self.load_vae_feat = load_vae_feat
        self.load_text_feat = load_text_feat
        self.resolution = resolution
        self.max_length = max_length
        self.caption_proportion = caption_proportion if caption_proportion is not None else {"prompt": 1.0}
        self.external_caption_suffixes = external_caption_suffixes
        self.external_clipscore_suffixes = external_clipscore_suffixes
        self.clip_thr = clip_thr
        self.clip_thr_temperature = clip_thr_temperature
        self.default_prompt = "prompt"
        self.img_extension = img_extension

        self.data_dirs = data_dir if isinstance(data_dir, list) else [data_dir]
        # self.meta_datas = [osp.join(data_dir, "meta_data.json") for data_dir in self.data_dirs]
        self.dataset = []
        for data_dir in self.data_dirs:
            meta_data = json.load(open(osp.join(data_dir, "meta_data.json")))
            self.dataset.extend([osp.join(data_dir, i) for i in meta_data["img_names"]])

        self.dataset = self.dataset * 2000
        self.logger.info(colored("Dataset is repeat 2000 times for toy dataset", "red", attrs=["bold"]))
        self.ori_imgs_nums = len(self)
        self.logger.info(f"Dataset samples: {len(self.dataset)}")

        self.logger.info(f"Loading external caption json from: original_filename{external_caption_suffixes}.json")
        self.logger.info(f"Loading external clipscore json from: original_filename{external_clipscore_suffixes}.json")
        self.logger.info(f"external caption clipscore threshold: {clip_thr}, temperature: {clip_thr_temperature}")
        self.logger.info(f"Text max token length: {self.max_length}")

    def getdata(self, idx):
        data = self.dataset[idx]
        img_extensions = [".jpg", ".png", ".jpeg", ".webp"]
        filename, ext = os.path.splitext(data)
        if ext in img_extensions:
            data = filename
            self.img_extension = ext
        self.key = data.split("/")[-1]
        info = {}
        with open(f"{data}.txt") as f:
            info[self.default_prompt] = f.readlines()[0].strip()

        # external json file
        for suffix in self.external_caption_suffixes:
            caption_json_path = f"{data}{suffix}.json"
            if os.path.exists(caption_json_path):
                try:
                    caption_json = lru_json_load(caption_json_path)
                except:
                    caption_json = {}
                if self.key in caption_json:
                    info.update(caption_json[self.key])

        caption_type, caption_clipscore = self.weighted_sample_clipscore(data, info)
        caption_type = caption_type if caption_type in info else self.default_prompt
        txt_fea = "" if info[caption_type] is None else info[caption_type]

        data_info = {
            "img_hw": torch.tensor([self.resolution, self.resolution], dtype=torch.float32),
            "aspect_ratio": torch.tensor(1.0),
        }

        if self.load_vae_feat:
            assert ValueError("Load VAE is not supported now")
        else:
            img = f"{data}{self.img_extension}"
            img = Image.open(img)
        if self.transform:
            img = self.transform(img)

        attention_mask = torch.ones(1, 1, self.max_length, dtype=torch.int16)  # 1x1xT
        if self.load_text_feat:
            npz_path = f"{self.key}.npz"
            txt_info = np.load(npz_path)
            txt_fea = torch.from_numpy(txt_info["caption_feature"])  # 1xTx4096
            if "attention_mask" in txt_info:
                attention_mask = torch.from_numpy(txt_info["attention_mask"])[None]
            # make sure the feature length are the same
            if txt_fea.shape[1] != self.max_length:
                txt_fea = torch.cat([txt_fea, txt_fea[:, -1:].repeat(1, self.max_length - txt_fea.shape[1], 1)], dim=1)
                attention_mask = torch.cat(
                    [attention_mask, torch.zeros(1, 1, self.max_length - attention_mask.shape[-1])], dim=-1
                )

        return (
            img,
            txt_fea,
            attention_mask.to(torch.int16),
            data_info,
            idx,
            caption_type,
            "",
            str(caption_clipscore),
        )

    def __getitem__(self, idx):
        for _ in range(10):
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = idx + 1
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.dataset)

    def weighted_sample_fix_prob(self):
        labels = list(self.caption_proportion.keys())
        weights = list(self.caption_proportion.values())
        sampled_label = random.choices(labels, weights=weights, k=1)[0]
        return sampled_label

    def weighted_sample_clipscore(self, data, info):
        labels = []
        weights = []
        fallback_label = None
        max_clip_score = float("-inf")

        for suffix in self.external_clipscore_suffixes:
            clipscore_json_path = f"{data}{suffix}.json"

            if os.path.exists(clipscore_json_path):
                try:
                    clipscore_json = lru_json_load(clipscore_json_path)
                except:
                    clipscore_json = {}
                if self.key in clipscore_json:
                    clip_scores = clipscore_json[self.key]

                    for caption_type, clip_score in clip_scores.items():
                        clip_score = float(clip_score)
                        if caption_type in info:
                            if clip_score >= self.clip_thr:
                                labels.append(caption_type)
                                weights.append(clip_score)

                            if clip_score > max_clip_score:
                                max_clip_score = clip_score
                                fallback_label = caption_type

        if not labels and fallback_label:
            return fallback_label, max_clip_score

        if not labels:
            return self.default_prompt, 0.0

        adjusted_weights = np.array(weights) ** (1.0 / max(self.clip_thr_temperature, 0.01))
        normalized_weights = adjusted_weights / np.sum(adjusted_weights)
        sampled_label = random.choices(labels, weights=normalized_weights, k=1)[0]
        # sampled_label = random.choices(labels, weights=[1]*len(weights), k=1)[0]
        index = labels.index(sampled_label)
        original_weight = weights[index]

        return sampled_label, original_weight


@DATASETS.register_module()
class SanaWebDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir="",
        meta_path=None,
        cache_dir="/cache/data/sana-webds-meta",
        max_shards_to_load=None,
        transform=None,
        resolution=256,
        load_vae_feat=False,
        load_text_feat=False,
        max_length=300,
        config=None,
        caption_proportion=None,
        sort_dataset=False,
        num_replicas=None,
        external_caption_suffixes=None,
        external_clipscore_suffixes=None,
        clip_thr=0.0,
        clip_thr_temperature=1.0,
        **kwargs,
    ):
        if external_caption_suffixes is None:
            external_caption_suffixes = []
        if external_clipscore_suffixes is None:
            external_clipscore_suffixes = []

        self.logger = (
            get_root_logger() if config is None else get_root_logger(osp.join(config.work_dir, "train_log.log"))
        )
        self.transform = transform if not load_vae_feat else None
        self.load_vae_feat = load_vae_feat
        self.load_text_feat = load_text_feat
        self.resolution = resolution
        self.max_length = max_length
        self.caption_proportion = caption_proportion if caption_proportion is not None else {"prompt": 1.0}
        self.external_caption_suffixes = external_caption_suffixes
        self.external_clipscore_suffixes = external_clipscore_suffixes
        self.clip_thr = clip_thr
        self.clip_thr_temperature = clip_thr_temperature
        self.default_prompt = "prompt"

        data_dirs = data_dir if isinstance(data_dir, list) else [data_dir]
        meta_paths = meta_path if isinstance(meta_path, list) else [meta_path] * len(data_dirs)
        self.meta_paths = []
        for data_path, meta_path in zip(data_dirs, meta_paths):
            self.data_path = osp.expanduser(data_path)
            self.meta_path = osp.expanduser(meta_path) if meta_path is not None else None

            _local_meta_path = osp.join(self.data_path, "wids-meta.json")
            if meta_path is None and osp.exists(_local_meta_path):
                self.logger.info(f"loading from {_local_meta_path}")
                self.meta_path = meta_path = _local_meta_path

            if meta_path is None:
                self.meta_path = osp.join(
                    osp.expanduser(cache_dir),
                    self.data_path.replace("/", "--") + f".max_shards:{max_shards_to_load}" + ".wdsmeta.json",
                )

            assert osp.exists(self.meta_path), f"meta path not found in [{self.meta_path}] or [{_local_meta_path}]"
            self.logger.info(f"[SimplyInternal] Loading meta information {self.meta_path}")
            self.meta_paths.append(self.meta_path)

        self._initialize_dataset(num_replicas, sort_dataset)

        self.logger.info(f"Loading external caption json from: original_filename{external_caption_suffixes}.json")
        self.logger.info(f"Loading external clipscore json from: original_filename{external_clipscore_suffixes}.json")
        self.logger.info(f"external caption clipscore threshold: {clip_thr}, temperature: {clip_thr_temperature}")
        self.logger.info(f"Text max token length: {self.max_length}")
        self.logger.warning(f"Sort the dataset: {sort_dataset}")

    def _initialize_dataset(self, num_replicas, sort_dataset):
        # uuid = abs(hash(self.meta_path)) % (10 ** 8)
        import hashlib

        uuid = hashlib.sha256(self.meta_path.encode()).hexdigest()[:8]
        if len(self.meta_paths) > 0:
            self.dataset = ShardListDatasetMulti(
                self.meta_paths,
                cache_dir=osp.expanduser(f"~/.cache/_wids_cache/{getpass.getuser()}-{uuid}"),
                sort_data_inseq=sort_dataset,
                num_replicas=num_replicas or dist.get_world_size(),
            )
        else:
            # TODO: tmp to ensure there is no bug
            self.dataset = ShardListDataset(
                self.meta_path,
                cache_dir=osp.expanduser(f"~/.cache/_wids_cache/{getpass.getuser()}-{uuid}"),
            )
        self.ori_imgs_nums = len(self)
        self.logger.info(f"{self.dataset.data_info}")

    def getdata(self, idx):
        data = self.dataset[idx]
        info = data[".json"]
        self.key = data["__key__"]
        dataindex_info = {
            "index": data["__index__"],
            "shard": "/".join(data["__shard__"].rsplit("/", 2)[-2:]),
            "shardindex": data["__shardindex__"],
        }

        # external json file
        for suffix in self.external_caption_suffixes:
            caption_json_path = data["__shard__"].replace(".tar", f"{suffix}.json")
            if os.path.exists(caption_json_path):
                try:
                    caption_json = lru_json_load(caption_json_path)
                except:
                    caption_json = {}
                if self.key in caption_json:
                    info.update(caption_json[self.key])

        caption_type, caption_clipscore = self.weighted_sample_clipscore(data, info)
        caption_type = caption_type if caption_type in info else self.default_prompt
        txt_fea = "" if info[caption_type] is None else info[caption_type]

        data_info = {
            "img_hw": torch.tensor([self.resolution, self.resolution], dtype=torch.float32),
            "aspect_ratio": torch.tensor(1.0),
        }

        if self.load_vae_feat:
            img = data[".npy"]
        else:
            img = data[".png"] if ".png" in data else data[".jpg"]
        if self.transform:
            img = self.transform(img)

        attention_mask = torch.ones(1, 1, self.max_length, dtype=torch.int16)  # 1x1xT
        if self.load_text_feat:
            npz_path = f"{self.key}.npz"
            txt_info = np.load(npz_path)
            txt_fea = torch.from_numpy(txt_info["caption_feature"])  # 1xTx4096
            if "attention_mask" in txt_info:
                attention_mask = torch.from_numpy(txt_info["attention_mask"])[None]
            # make sure the feature length are the same
            if txt_fea.shape[1] != self.max_length:
                txt_fea = torch.cat([txt_fea, txt_fea[:, -1:].repeat(1, self.max_length - txt_fea.shape[1], 1)], dim=1)
                attention_mask = torch.cat(
                    [attention_mask, torch.zeros(1, 1, self.max_length - attention_mask.shape[-1])], dim=-1
                )

        return (
            img,
            txt_fea,
            attention_mask.to(torch.int16),
            data_info,
            idx,
            caption_type,
            dataindex_info,
            str(caption_clipscore),
        )

    def __getitem__(self, idx):
        for _ in range(10):
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = idx + 1
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.dataset)

    def weighted_sample_fix_prob(self):
        labels = list(self.caption_proportion.keys())
        weights = list(self.caption_proportion.values())
        sampled_label = random.choices(labels, weights=weights, k=1)[0]
        return sampled_label

    def weighted_sample_clipscore(self, data, info):
        labels = []
        weights = []
        fallback_label = None
        max_clip_score = float("-inf")

        for suffix in self.external_clipscore_suffixes:
            clipscore_json_path = data["__shard__"].replace(".tar", f"{suffix}.json")

            if os.path.exists(clipscore_json_path):
                try:
                    clipscore_json = lru_json_load(clipscore_json_path)
                except:
                    clipscore_json = {}
                if self.key in clipscore_json:
                    clip_scores = clipscore_json[self.key]

                    for caption_type, clip_score in clip_scores.items():
                        clip_score = float(clip_score)
                        if caption_type in info:
                            if clip_score >= self.clip_thr:
                                labels.append(caption_type)
                                weights.append(clip_score)

                            if clip_score > max_clip_score:
                                max_clip_score = clip_score
                                fallback_label = caption_type

        if not labels and fallback_label:
            return fallback_label, max_clip_score

        if not labels:
            return self.default_prompt, 0.0

        adjusted_weights = np.array(weights) ** (1.0 / max(self.clip_thr_temperature, 0.01))
        normalized_weights = adjusted_weights / np.sum(adjusted_weights)
        sampled_label = random.choices(labels, weights=normalized_weights, k=1)[0]
        # sampled_label = random.choices(labels, weights=[1]*len(weights), k=1)[0]
        index = labels.index(sampled_label)
        original_weight = weights[index]

        return sampled_label, original_weight

    def get_data_info(self, idx):
        try:
            data = self.dataset[idx]
            info = data[".json"]
            key = data["__key__"]
            version = info.get("version", "others")
            return {"height": info["height"], "width": info["width"], "version": version, "key": key}
        except Exception as e:
            print(f"Error details: {str(e)}")
            return None


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from diffusion.data.transforms import get_transform

    image_size = 1024  # 256
    transform = get_transform("default_train", image_size)
    train_dataset = SanaWebDataset(
        data_dir="debug_data_train/vaef32c32/debug_data",
        resolution=image_size,
        transform=transform,
        max_length=300,
        load_vae_feat=True,
        num_replicas=1,
    )
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)

    for data in dataloader:
        img, txt_fea, attention_mask, data_info = data
        print(txt_fea)
        break
