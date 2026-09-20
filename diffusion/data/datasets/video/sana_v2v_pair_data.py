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

"""Manifest-backed paired-video dataset for SANA-Streaming training."""

import io
import json
import os.path as osp
import random
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

import imageio.v3 as iio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from diffusion.data.builder import DATASETS
from diffusion.data.datasets import utils as dataset_utils
from diffusion.data.transforms import ResizeCrop, ToTensorVideo, get_closest_ratio
from diffusion.utils.logger import get_root_logger

_REQUIRED_MANIFEST_FIELDS = {
    "id",
    "shard",
    "source_member",
    "target_member",
    "prompt",
    "width",
    "height",
}


def _safe_relative_path(value: str, field: str) -> str:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a dataset-relative path, got {value!r}")
    return path.as_posix()


def _read_manifest(path: str) -> list[dict]:
    rows = []
    seen_ids = set()
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = sorted(_REQUIRED_MANIFEST_FIELDS.difference(row))
            if missing:
                raise ValueError(f"{path}:{line_number} is missing fields: {missing}")
            sample_id = str(row["id"])
            if sample_id in seen_ids:
                raise ValueError(f"{path}:{line_number} repeats sample id {sample_id!r}")
            seen_ids.add(sample_id)
            row["id"] = sample_id
            row["shard"] = _safe_relative_path(row["shard"], "shard")
            row["source_member"] = _safe_relative_path(row["source_member"], "source_member")
            row["target_member"] = _safe_relative_path(row["target_member"], "target_member")
            rows.append(row)
    return rows


@lru_cache(maxsize=32)
def _open_zip(path: str) -> ZipFile:
    return ZipFile(path, "r")


def _decode_video(payload: bytes, member_name: str) -> np.ndarray:
    if member_name.lower().endswith(".npy"):
        frames = np.load(io.BytesIO(payload), allow_pickle=False)
    else:
        frames = iio.imread(payload, plugin="pyav", extension=Path(member_name).suffix or ".mp4")

    frames = np.asarray(frames)
    if frames.ndim == 3:
        frames = np.repeat(frames[..., None], 3, axis=-1)
    if frames.ndim != 4:
        raise ValueError(f"Expected a THWC video, got shape {frames.shape} for {member_name}")
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    if frames.shape[-1] != 3:
        raise ValueError(f"Expected 3 color channels, got shape {frames.shape} for {member_name}")
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return frames


def _resolve_aspect_ratios(name: str) -> dict:
    ratios = getattr(dataset_utils, name, None)
    if not isinstance(ratios, dict):
        raise ValueError(f"Unknown aspect_ratio_type: {name!r}")
    return ratios


@DATASETS.register_module()
class SanaV2VPairDataset(Dataset):
    """Load aligned source/target videos from self-contained ZIP shards.

    Each data directory must contain ``manifest.jsonl``. Manifest paths are
    relative to that directory, so a Hugging Face snapshot can be relocated
    without regenerating metadata. The returned tuple intentionally matches
    the public video trainer: target video is item 0 and source video is item 8.
    """

    def __init__(
        self,
        data_dir,
        transform=None,
        load_vae_feat=False,
        load_text_feat=False,
        config=None,
        caption_proportion=None,
        sort_dataset=False,
        num_frames=81,
        aspect_ratio_type="ASPECT_RATIO_VIDEO_720_MS_DIV32",
        manifest_name="manifest.jsonl",
        max_retries=20,
        **kwargs,
    ):
        del transform, caption_proportion, kwargs
        if load_vae_feat:
            raise ValueError("SanaV2VPairDataset decodes raw paired videos; set data.load_vae_feat=false.")
        if load_text_feat:
            raise ValueError("SanaV2VPairDataset returns raw prompts; set data.load_text_feat=false.")

        self.logger = (
            get_root_logger() if config is None else get_root_logger(osp.join(config.work_dir, "train_log.log"))
        )
        self.load_vae_feat = False
        self.load_text_feat = False
        self.num_frames = int(num_frames)
        self.max_retries = int(max_retries)
        self.max_length = int(getattr(getattr(config, "text_encoder", None), "model_max_length", 300))
        self.aspect_ratio = _resolve_aspect_ratios(aspect_ratio_type)

        if isinstance(data_dir, dict):
            data_dirs = data_dir
        elif isinstance(data_dir, str):
            data_dirs = {"default": data_dir}
        else:
            data_dirs = {f"default_{index}": path for index, path in enumerate(data_dir or [])}

        self.dataset = []
        for dataset_name, root in data_dirs.items():
            root = osp.abspath(osp.expanduser(str(root)))
            manifest_path = osp.join(root, manifest_name)
            if not osp.isfile(manifest_path):
                raise FileNotFoundError(f"V2V manifest not found: {manifest_path}")
            for row in _read_manifest(manifest_path):
                row["root"] = root
                row["dataset_name"] = str(dataset_name)
                row["shard_path"] = osp.join(root, row["shard"])
                if not osp.isfile(row["shard_path"]):
                    raise FileNotFoundError(f"V2V shard not found: {row['shard_path']}")
                self.dataset.append(row)

        if sort_dataset:
            self.dataset.sort(key=lambda row: row["id"])
        if not self.dataset:
            raise ValueError("SanaV2VPairDataset found no samples.")

        self.ori_imgs_nums = len(self.dataset)
        self.shuffle_dataset = False
        self.ratio_index = {float(key): [] for key in self.aspect_ratio}
        self.ratio_nums = {float(key): 0 for key in self.aspect_ratio}
        for index, row in enumerate(self.dataset):
            _, ratio = get_closest_ratio(float(row["height"]), float(row["width"]), self.aspect_ratio)
            self.ratio_index[ratio].append(index)
            self.ratio_nums[ratio] += 1

        self.logger.info(
            f"SanaV2VPairDataset loaded {len(self.dataset)} pairs from {len(data_dirs)} manifest directory/directories."
        )

    def _load_pair(self, row: dict) -> tuple[np.ndarray, np.ndarray]:
        archive = _open_zip(row["shard_path"])
        source = _decode_video(archive.read(row["source_member"]), row["source_member"])
        target = _decode_video(archive.read(row["target_member"]), row["target_member"])
        frame_count = min(len(source), len(target))
        if frame_count < self.num_frames:
            raise ValueError(
                f"Pair {row['id']} has {frame_count} aligned frames, fewer than requested {self.num_frames}."
            )
        return source[: self.num_frames], target[: self.num_frames]

    def _transform_pair(self, source: np.ndarray, target: np.ndarray, size: tuple[int, int]):
        transform = T.Compose(
            [
                ToTensorVideo(),
                ResizeCrop(size),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        source_tensor = torch.from_numpy(np.ascontiguousarray(source)).permute(0, 3, 1, 2)
        target_tensor = torch.from_numpy(np.ascontiguousarray(target)).permute(0, 3, 1, 2)
        return transform(source_tensor), transform(target_tensor)

    def getdata(self, index: int):
        row = self.dataset[index]
        height, width = float(row["height"]), float(row["width"])
        closest_size, closest_ratio = get_closest_ratio(height, width, self.aspect_ratio)
        size = tuple(int(value) for value in closest_size)
        source, target = self._load_pair(row)
        source, target = self._transform_pair(source, target, size)

        data_info = {
            "cache_key": f"{row['dataset_name']}/{row['id']}",
            "zip_file": row["shard_path"],
            "key": row["id"],
            "dataset_name": row["dataset_name"],
            "img_hw": torch.tensor([height, width], dtype=torch.float32),
            "aspect_ratio": closest_ratio,
        }
        attention_mask = torch.ones(1, 1, self.max_length, dtype=torch.int16)
        return (
            target,
            str(row["prompt"]),
            attention_mask,
            data_info,
            index,
            "prompt",
            {"height": height, "width": width},
            0.0,
            source,
        )

    def __getitem__(self, index: int):
        last_error = None
        for _ in range(self.max_retries):
            try:
                return self.getdata(index)
            except Exception as error:
                last_error = error
                self.logger.warning(f"Failed to load V2V pair {self.dataset[index]['id']}: {error}")
                row = self.dataset[index]
                _, closest_ratio = get_closest_ratio(float(row["height"]), float(row["width"]), self.aspect_ratio)
                index = random.choice(self.ratio_index[closest_ratio])
        raise RuntimeError(f"Failed to load a V2V pair after {self.max_retries} attempts") from last_error

    def __len__(self):
        return len(self.dataset)

    def get_data_info(self, index: int):
        row = self.dataset[index]
        height, width = float(row["height"]), float(row["width"])
        _, closest_ratio = get_closest_ratio(height, width, self.aspect_ratio)
        return {
            "height": height,
            "width": width,
            "key": row["id"],
            "index": index,
            "zip_file": row["shard_path"],
            "closest_ratio": closest_ratio,
        }
