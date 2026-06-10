# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import os.path as osp
import random
import warnings
from functools import lru_cache
from glob import glob
from zipfile import ZipFile

import numpy as np
import torch
from torch.utils.data import Dataset

from diffusion.data.builder import DATASETS
from diffusion.utils.cam_utils import compute_raymap


def _as_data_dirs(data_dir):
    if isinstance(data_dir, dict):
        return data_dir
    if isinstance(data_dir, str):
        return {"default": data_dir}
    return {f"default_{idx}": path for idx, path in enumerate(data_dir or [])}


def _relative_poses(poses: np.ndarray) -> np.ndarray:
    if poses.shape[0] == 0:
        return poses
    first_inv = np.linalg.inv(poses[0])
    rel = first_inv[None] @ poses
    rel[0] = np.eye(4, dtype=np.float32)
    return rel.astype(np.float32)


@DATASETS.register_module()
class SanaWMZipLatentDataset(Dataset):
    """Read SANA-WM raw zip metadata paired with LTX VAE latent-cache zips.

    Raw data directories contain ``*.zip`` files with ``<key>.json`` entries.
    ``vae_cache_dir`` contains zips with matching basenames and ``<key>.npz``
    entries storing ``z`` latents in ``(C, T, H, W)`` layout. Optional camera
    sidecars live next to raw zips as ``<raw_zip_stem>_camera.npz`` and
    ``<raw_zip_stem>_metric_scale_stats.json``.
    """

    def __init__(
        self,
        data_dir,
        vae_cache_dir: str,
        transform=None,
        resolution: int = 720,
        num_frames: int | None = None,
        length: int | None = None,
        min_latent_file_size: int = 10 * 1024 * 1024,
        caption_proportion: dict | None = None,
        external_caption_suffixes: list[str] | None = None,
        external_data_filter: dict | None = None,
        data_repeat: dict | int | None = None,
        sort_dataset: bool = False,
        shuffle_dataset: bool = False,
        return_chunk_plucker: bool = False,
        vae_ratio: tuple[int, int] = (8, 32),
        cam_sample_strategy: str = "last",
        **_: object,
    ) -> None:
        del transform
        if vae_cache_dir is None:
            raise ValueError("SanaWMZipLatentDataset requires vae_cache_dir.")

        self.resolution = int(resolution)
        self.num_frames = None if num_frames is None else int(num_frames)
        self.min_latent_file_size = int(min_latent_file_size)
        self.caption_proportion = caption_proportion if caption_proportion is not None else {"prompt": 1.0}
        self.external_caption_suffixes = list(external_caption_suffixes or [])
        self.external_data_filter = external_data_filter or {}
        self.data_repeat = data_repeat or {}
        self.default_prompt = "prompt"
        self.load_vae_feat = True
        self.load_text_feat = False
        self.shuffle_dataset = bool(shuffle_dataset)
        self.return_chunk_plucker = bool(return_chunk_plucker)
        self.vae_time_stride = int(vae_ratio[0])
        self.vae_spatial_stride = int(vae_ratio[1])
        self.cam_sample_strategy = str(cam_sample_strategy)
        if self.cam_sample_strategy not in {"first", "last"}:
            raise ValueError(f"Invalid cam_sample_strategy: {self.cam_sample_strategy}")
        self.aspect_ratio = {"1.00": [self.resolution, self.resolution]}
        self.dataset = []

        cache_root = osp.abspath(osp.expanduser(vae_cache_dir))
        for dataset_name, raw_dir in _as_data_dirs(data_dir).items():
            raw_dir = osp.abspath(osp.expanduser(raw_dir))
            start = len(self.dataset)
            for raw_zip in sorted(glob(osp.join(raw_dir, "*.zip"))):
                cache_zip = osp.join(cache_root, osp.basename(raw_zip))
                if not osp.exists(cache_zip):
                    continue
                camera_npz = raw_zip.replace(".zip", "_camera.npz")
                self._add_zip_items(dataset_name, raw_zip, cache_zip, camera_npz)
            repeat = self._repeat_for_dataset(dataset_name)
            if repeat > 1:
                items = self.dataset[start:]
                self.dataset.extend(dict(item) for _ in range(repeat - 1) for item in items)

        if sort_dataset:
            self.dataset.sort(key=lambda item: (item["dataset_name"], item["key"]))
        elif self.shuffle_dataset:
            random.shuffle(self.dataset)

        self.ori_imgs_nums = len(self.dataset)
        if length is not None and int(length) > 0:
            self.dataset = self.dataset[: int(length)]
        self.ratio_nums = {"1.00": len(self.dataset)}

    @staticmethod
    def read_zip_entry(path: str, name: str) -> bytes:
        with ZipFile(path, "r") as zip_file:
            return zip_file.read(name)

    @staticmethod
    @lru_cache(8)
    def load_camera_sidecar(path: str):
        if not osp.exists(path):
            return None
        return np.load(path, allow_pickle=False)

    @staticmethod
    @lru_cache(32)
    def load_json_sidecar(path: str) -> dict:
        if not osp.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            return {str(item["key"]): item for item in data if isinstance(item, dict) and "key" in item}
        return {}

    @staticmethod
    @lru_cache(32)
    def load_jsonl_sidecar(path: str) -> dict:
        if not osp.exists(path):
            return {}
        out = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict) and "key" in item:
                    out[str(item["key"])] = item
        return out

    def _add_zip_items(self, dataset_name: str, raw_zip: str, cache_zip: str, camera_npz: str) -> None:
        with ZipFile(raw_zip, "r") as raw, ZipFile(cache_zip, "r") as cache:
            cache_infos = {
                osp.splitext(info.filename)[0]: info
                for info in cache.infolist()
                if info.filename.endswith(".npz") and info.file_size >= self.min_latent_file_size
            }
            for info in raw.infolist():
                if not info.filename.endswith(".json"):
                    continue
                key = osp.splitext(info.filename)[0]
                if key not in cache_infos:
                    continue
                if not self._passes_external_filters(dataset_name, raw_zip, key):
                    continue
                self.dataset.append(
                    {
                        "dataset_name": dataset_name,
                        "key": key,
                        "json_name": info.filename,
                        "raw_zip": raw_zip,
                        "cache_zip": cache_zip,
                        "cache_name": cache_infos[key].filename,
                        "camera_npz": camera_npz,
                    }
                )

    def _repeat_for_dataset(self, dataset_name: str) -> int:
        repeat = self.data_repeat
        if isinstance(repeat, dict):
            repeat = repeat.get(dataset_name, 1)
        try:
            return max(1, int(repeat))
        except (TypeError, ValueError):
            return 1

    def _filter_specs_for_dataset(self, dataset_name: str) -> dict:
        specs = self.external_data_filter
        if not isinstance(specs, dict):
            return {}
        dataset_specs = specs.get(dataset_name)
        if isinstance(dataset_specs, dict):
            return dataset_specs
        if any(isinstance(value, dict) and "min" not in value and "max" not in value for value in specs.values()):
            return {}
        return specs

    @staticmethod
    def _numeric_score(value) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, list):
            numbers = [float(item) for item in value if isinstance(item, (int, float))]
            return sum(numbers) / len(numbers) if numbers else None
        if isinstance(value, dict):
            preferred = (
                "score",
                "vmafmotion_score",
                "dover_score",
                "quality_pass",
                "entity_count",
                "unimatch_flow_score",
                "median",
            )
            for key in preferred:
                if key in value:
                    score = SanaWMZipLatentDataset._numeric_score(value[key])
                    if score is not None:
                        return score
            for item in value.values():
                score = SanaWMZipLatentDataset._numeric_score(item)
                if score is not None:
                    return score
        return None

    def _passes_external_filters(self, dataset_name: str, raw_zip: str, key: str) -> bool:
        for suffix, bounds in self._filter_specs_for_dataset(dataset_name).items():
            if not isinstance(bounds, dict):
                continue
            sidecar_base = osp.splitext(raw_zip)[0] + suffix
            filter_data = self.load_json_sidecar(f"{sidecar_base}.json")
            if not filter_data:
                filter_data = self.load_jsonl_sidecar(f"{sidecar_base}.jsonl")
            score = self._numeric_score(filter_data.get(key))
            if score is None:
                return False
            min_value = float(bounds.get("min", float("-inf")))
            max_value = float(bounds.get("max", float("inf")))
            if score < min_value or score > max_value:
                return False
        return True

    def _metric_scale_for_item(self, item: dict) -> float:
        scale_data = self.load_json_sidecar(item["raw_zip"].replace(".zip", "_metric_scale_stats.json"))
        score = self._numeric_score(scale_data.get(item["key"]))
        if score is None or not np.isfinite(score):
            return 1.0
        return float(score)

    def _read_latent(self, item: dict) -> torch.Tensor:
        raw = self.read_zip_entry(item["cache_zip"], item["cache_name"])
        npz = np.load(io.BytesIO(raw), allow_pickle=False)
        latent = npz["z"] if hasattr(npz, "files") else npz
        latent = torch.from_numpy(np.asarray(latent)).float()
        if self.num_frames is not None and latent.shape[1] > self.num_frames:
            latent = latent[:, : self.num_frames]
        return latent

    def _read_info(self, item: dict) -> dict:
        raw = self.read_zip_entry(item["raw_zip"], item["json_name"])
        return json.loads(raw.decode("utf-8"))

    def _add_external_captions(self, item: dict, info: dict) -> None:
        for suffix in self.external_caption_suffixes:
            sidecar_base = osp.splitext(item["raw_zip"])[0] + suffix
            caption_data = self.load_json_sidecar(f"{sidecar_base}.json")
            if not caption_data:
                caption_data = self.load_jsonl_sidecar(f"{sidecar_base}.jsonl")
            external_info = caption_data.get(item["key"])
            if external_info is None:
                continue
            caption_key = suffix.replace(".", "_")
            if isinstance(external_info, str):
                info[caption_key] = external_info
            elif isinstance(external_info, dict):
                if self.default_prompt in external_info:
                    info[caption_key] = external_info[self.default_prompt]
                else:
                    for value in external_info.values():
                        if isinstance(value, str):
                            info[caption_key] = value
                            break

    def _caption_weights_for_dataset(self, dataset_name: str) -> dict:
        weights = self.caption_proportion
        if not isinstance(weights, dict):
            return {self.default_prompt: 1.0}
        dataset_weights = weights.get(dataset_name)
        if isinstance(dataset_weights, dict):
            return dataset_weights
        if any(isinstance(value, dict) for value in weights.values()):
            return {self.default_prompt: 1.0}
        return weights

    def _select_caption(self, item: dict, info: dict) -> tuple[str, str]:
        weights = self._caption_weights_for_dataset(item["dataset_name"])
        available = [
            (caption_type, float(weight))
            for caption_type, weight in weights.items()
            if caption_type in info and info[caption_type] is not None
        ]
        if not available:
            return self.default_prompt, str(info.get(self.default_prompt) or "")
        caption_types, caption_weights = zip(*available)
        if sum(caption_weights) <= 0:
            caption_type = self.default_prompt if self.default_prompt in caption_types else caption_types[0]
        else:
            caption_type = random.choices(caption_types, weights=caption_weights, k=1)[0]
        return caption_type, str(info.get(caption_type) or "")

    def _read_camera_data(
        self,
        item: dict,
        info: dict,
        latent_t: int,
        latent_h: int,
        latent_w: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        camera = self.load_camera_sidecar(item["camera_npz"])
        if camera is None or item["key"] not in set(camera["ids"].tolist()):
            poses = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(latent_t, 1, 1)
            intrinsics = torch.tensor(
                [latent_w, latent_h, latent_w * 0.5, latent_h * 0.5],
                dtype=torch.float32,
            ).repeat(latent_t, 1)
            chunk_plucker = self._build_chunk_plucker(poses, intrinsics, latent_h, latent_w)
            camera_conditions = torch.cat([poses.reshape(latent_t, 16), intrinsics], dim=1)
            return camera_conditions, chunk_plucker

        ids = camera["ids"].tolist()
        cam_idx = ids.index(item["key"])
        start, count = camera["ranges"][cam_idx]
        raw_poses = camera["pose"][start : start + int(count)].astype(np.float32)
        raw_intr = camera["intrinsics"][start : start + int(count)].astype(np.float32)
        if self.num_frames is not None:
            raw_poses = raw_poses[: self.num_frames]
            raw_intr = raw_intr[: self.num_frames]
        raw_poses[:, :3, 3] *= self._metric_scale_for_item(item)

        poses = torch.from_numpy(_relative_poses(raw_poses)).float()
        intrinsics = torch.from_numpy(raw_intr.copy()).float()
        width = float(info.get("width", self.resolution))
        height = float(info.get("height", self.resolution))
        intrinsics[:, [0, 2]] *= float(latent_w) / max(width, 1.0)
        intrinsics[:, [1, 3]] *= float(latent_h) / max(height, 1.0)

        if self.cam_sample_strategy == "last":
            time_indices = torch.arange(0, poses.shape[0], self.vae_time_stride)
        else:
            time_indices = torch.arange(0, poses.shape[0], self.vae_time_stride) - self.vae_time_stride + 1
            time_indices[0] = 0
        if len(time_indices) > latent_t:
            time_indices = time_indices[:latent_t]
        if len(time_indices) < latent_t:
            pad = time_indices[-1:].repeat(latent_t - len(time_indices))
            time_indices = torch.cat([time_indices, pad])

        poses_sub = poses[time_indices]
        intr_sub = intrinsics[time_indices]
        camera_conditions = torch.cat([poses_sub.reshape(latent_t, 16), intr_sub], dim=1)
        chunk_plucker = self._build_chunk_plucker(poses, intrinsics, latent_h, latent_w, time_indices)
        return camera_conditions, chunk_plucker

    def _build_chunk_plucker(
        self,
        poses: torch.Tensor,
        intrinsics: torch.Tensor,
        latent_h: int,
        latent_w: int,
        time_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if not self.return_chunk_plucker:
            return None
        if time_indices is None:
            time_indices = torch.arange(poses.shape[0])
        if self.cam_sample_strategy == "last":
            chunk_starts = time_indices - (self.vae_time_stride - 1)
        else:
            chunk_starts = time_indices

        plucker_chunks = []
        for chunk_start in chunk_starts:
            start = max(0, int(chunk_start))
            end = start + self.vae_time_stride
            chunk_poses = poses[start:end]
            chunk_intr = intrinsics[start:end]
            if chunk_poses.shape[0] < self.vae_time_stride:
                pad_len = self.vae_time_stride - chunk_poses.shape[0]
                chunk_poses = torch.cat([chunk_poses, chunk_poses[-1:].repeat(pad_len, 1, 1)], dim=0)
                chunk_intr = torch.cat([chunk_intr, chunk_intr[-1:].repeat(pad_len, 1)], dim=0)
            plucker = compute_raymap(chunk_intr, chunk_poses, latent_h, latent_w, use_plucker=True)
            plucker = plucker.permute(0, 3, 1, 2).reshape(-1, latent_h, latent_w)
            plucker_chunks.append(plucker)
        if not plucker_chunks:
            return torch.zeros(self.vae_time_stride * 6, 0, latent_h, latent_w, dtype=poses.dtype)
        return torch.stack(plucker_chunks).permute(1, 0, 2, 3).contiguous()

    def getdata(self, idx: int):
        item = self.dataset[idx]
        latent = self._read_latent(item)
        info = self._read_info(item)
        self._add_external_captions(item, info)
        camera, chunk_plucker = self._read_camera_data(
            item,
            info,
            int(latent.shape[1]),
            int(latent.shape[2]),
            int(latent.shape[3]),
        )
        caption_type, prompt = self._select_caption(item, info)
        data_info = {
            "img_hw": torch.tensor(
                [float(info.get("height", self.resolution)), float(info.get("width", self.resolution))],
                dtype=torch.float32,
            ),
            "aspect_ratio": torch.tensor([1.0], dtype=torch.float32),
            "cache_key": f"{item['dataset_name']}/{item['key']}",
            "key": item["key"],
            "zip_file": item["raw_zip"],
            "caption_type": caption_type,
        }
        ret = (latent, prompt, torch.ones(1, 1, 1, dtype=torch.int16), data_info, idx, prompt, camera)
        if self.return_chunk_plucker:
            ret = ret + (chunk_plucker,)
        return ret

    def __getitem__(self, idx: int):
        for _ in range(100):
            try:
                return self.getdata(idx)
            except Exception as exc:
                warnings.warn(f"{self.__class__.__name__}.getdata({idx}) failed: {exc}", stacklevel=2)
                idx = random.randrange(len(self.dataset))
        raise RuntimeError("Too many bad SANA-WM zip latent samples.")

    def __len__(self) -> int:
        return len(self.dataset)
