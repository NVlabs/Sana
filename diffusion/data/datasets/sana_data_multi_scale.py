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
import os
import random
import traceback

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from diffusion.data.builder import DATASETS
from diffusion.data.datasets.sana_data import SanaWebDataset
from diffusion.data.datasets.utils import *
from diffusion.data.datasets.utils import ASPECT_RATIO_2048, ASPECT_RATIO_2880
from diffusion.data.transforms import get_closest_ratio
from diffusion.data.wids import lru_json_load


@DATASETS.register_module()
class SanaWebDatasetMS(SanaWebDataset):
    def __init__(
        self,
        data_dir="",
        meta_path=None,
        cache_dir="/cache/data/sana-webds-meta",
        max_shards_to_load=None,
        transform=None,
        resolution=256,
        aspect_ratio_type=None,
        sample_subset=None,
        load_vae_feat=False,
        load_text_feat=False,
        input_size=32,
        patch_size=2,
        max_length=300,
        config=None,
        caption_proportion=None,
        sort_dataset=False,
        num_replicas=None,
        caption_selection_type="clipscore",  # clipscore, proportion
        external_caption_suffixes=None,
        external_clipscore_suffixes=None,
        clip_thr=0.0,
        clip_thr_temperature=1.0,
        vae_downsample_rate=32,
        **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            meta_path=meta_path,
            cache_dir=cache_dir,
            max_shards_to_load=max_shards_to_load,
            transform=transform,
            resolution=resolution,
            sample_subset=sample_subset,
            load_vae_feat=load_vae_feat,
            load_text_feat=load_text_feat,
            input_size=input_size,
            patch_size=patch_size,
            max_length=max_length,
            config=config,
            caption_proportion=caption_proportion,
            caption_selection_type=caption_selection_type,
            sort_dataset=sort_dataset,
            num_replicas=num_replicas,
            external_caption_suffixes=external_caption_suffixes,
            external_clipscore_suffixes=external_clipscore_suffixes,
            clip_thr=clip_thr,
            clip_thr_temperature=clip_thr_temperature,
            vae_downsample_rate=32,
            **kwargs,
        )
        self.base_size = resolution
        self.aspect_ratio = eval(aspect_ratio_type)  # base aspect ratio
        self.aspect_ratio_type = aspect_ratio_type
        self.ratio_index = {}
        self.ratio_nums = {}
        self.interpolate_model = InterpolationMode.BICUBIC
        self.interpolate_model = (
            InterpolationMode.BICUBIC
            if self.aspect_ratio not in [ASPECT_RATIO_2048, ASPECT_RATIO_2880]
            else InterpolationMode.LANCZOS
        )

        for k, v in self.aspect_ratio.items():
            self.ratio_index[float(k)] = []
            self.ratio_nums[float(k)] = 0

        self.vae_downsample_rate = vae_downsample_rate

    def __getitem__(self, idx):
        for _ in range(10):
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                traceback_str = traceback.format_exc()
                print(f"Error details: {str(e)}\n" f"Traceback:\n{traceback_str}")
                idx = random.choice(self.ratio_index[self.closest_ratio])
        raise RuntimeError("Too many bad data.")

    def getdata(self, idx):
        data = self.dataset[idx]
        info = data[".json"]
        self.key = data["__key__"]
        dataindex_info = {
            "key": data["__key__"],
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
                    if self.default_prompt in caption_json[self.key]:
                        info.update({suffix.replace(".", "_"): caption_json[self.key][self.default_prompt]})
                    else:
                        info.update(caption_json[self.key])

        data_info = {}
        ori_h, ori_w = info["height"], info["width"]

        # Calculate the closest aspect ratio and resize & crop image[w, h]
        closest_size, closest_ratio = get_closest_ratio(ori_h, ori_w, self.aspect_ratio)
        closest_size = list(map(lambda x: int(x), closest_size))
        self.closest_ratio = closest_ratio

        data_info["img_hw"] = torch.tensor([ori_h, ori_w], dtype=torch.float32)
        data_info["aspect_ratio"] = closest_ratio

        if self.caption_selection_type == "clipscore":
            caption_type, caption_clipscore = self.weighted_sample_clipscore(data, info)
        elif self.caption_selection_type == "proportion":
            caption_type = self.weighted_sample_fix_prob()
        else:
            raise ValueError(f"Invalid caption selection type: {self.caption_selection_type}")

        if caption_type not in info:
            self.logger.warning(f"Caption [{caption_type}] is not in info, data path: {data['__shard__']}")
            txt_fea = info[self.default_prompt]
        elif info[caption_type] is None:
            self.logger.warning(f"Caption type info[{caption_type}] is None, data path: {data['__shard__']}")
            txt_fea = ""
        else:
            txt_fea = info[caption_type]

        if self.load_vae_feat:
            img = data[".npy"]
            if len(img.shape) == 4 and img.shape[0] == 1:
                img = img[0]
            h, w = (img.shape[1], img.shape[2])
            assert h == int(closest_size[0] // self.vae_downsample_rate) and w == int(
                closest_size[1] // self.vae_downsample_rate
            ), f"h: {h}, w: {w}, ori_hw: {closest_size}, data_info: {dataindex_info}"
        else:
            img = data[".png"] if ".png" in data else data[".jpg"]
            if closest_size[0] / ori_h > closest_size[1] / ori_w:
                resize_size = closest_size[0], int(ori_w * closest_size[0] / ori_h)
            else:
                resize_size = int(ori_h * closest_size[1] / ori_w), closest_size[1]
            self.transform = T.Compose(
                [
                    T.Lambda(lambda img: img.convert("RGB")),
                    T.Resize(resize_size, interpolation=self.interpolate_model),  # Image.BICUBIC
                    T.CenterCrop(closest_size),
                    T.ToTensor(),
                    T.Normalize([0.5], [0.5]),
                ]
            )
        if idx not in self.ratio_index[closest_ratio]:
            self.ratio_index[closest_ratio].append(idx)

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
        )

    def __len__(self):
        return len(self.dataset)


@DATASETS.register_module()
class DummyDatasetMS(SanaWebDatasetMS):
    def __init__(self, **kwargs):
        self.base_size = kwargs["resolution"]
        self.aspect_ratio = eval(kwargs.pop("aspect_ratio_type"))  # base aspect ratio
        self.ratio_index = {}
        self.ratio_nums = {}
        self.interpolate_model = InterpolationMode.BICUBIC
        self.interpolate_model = (
            InterpolationMode.BICUBIC
            if self.aspect_ratio not in [ASPECT_RATIO_2048, ASPECT_RATIO_2880]
            else InterpolationMode.LANCZOS
        )

        for k, v in self.aspect_ratio.items():
            self.ratio_index[float(k)] = []
            self.ratio_nums[float(k)] = 0

        self.ori_imgs_nums = 1_000_000
        self.height = 384
        self.width = 672

    def __getitem__(self, idx):
        img = torch.randn((3, self.height, self.width))
        txt_fea = "The image depicts a young woman standing in the middle of a street, leaning against a silver car. She is dressed in a stylish outfit consisting of a blue blouse and black pants. Her hair is long and dark, and she is looking directly at the camera with a confident expression. The street is lined with colorful buildings, and the trees have autumn leaves, suggesting the season is fall. The lighting is warm, with sunlight casting long shadows on the street. There are a few people in the background, and the overall atmosphere is vibrant and lively."
        attention_mask = torch.ones(1, 1, 300, dtype=torch.int16)  # 1x1xT
        data_info = {"img_hw": torch.tensor([816.0, 1456.0]), "aspect_ratio": 0.57}
        idx = 2500
        caption_type = self.default_prompt
        dataindex_info = {"index": 2500, "shard": "data_for_test_after_change/00000000.tar", "shardindex": 2500}
        return img, txt_fea, attention_mask, data_info, idx, caption_type, dataindex_info

    def __len__(self):
        return self.ori_imgs_nums

    def get_data_info(self, idx):
        return {"height": self.height, "width": self.width, "version": "1.0", "key": "dummpy_key"}


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from diffusion.data.datasets.utils import ASPECT_RATIO_1024
    from diffusion.data.transforms import get_transform

    image_size = 256
    transform = get_transform("default_train", image_size)
    data_dir = ["data/debug_data_train/debug_data"]
    for data_path in data_dir:
        train_dataset = SanaWebDatasetMS(data_dir=data_path, resolution=image_size, transform=transform, max_length=300)
        dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)

        for data in tqdm(dataloader):
            break
        print(dataloader.dataset.index_info)
