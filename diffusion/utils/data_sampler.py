# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import random
from copy import deepcopy
from random import choice, shuffle
from typing import Sequence

from torch.utils.data import BatchSampler, Dataset, Sampler

from diffusion.utils.logger import get_root_logger


class AspectRatioBatchSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(
        self,
        sampler: Sampler,
        dataset: Dataset,
        batch_size: int,
        aspect_ratios: dict,
        drop_last: bool = False,
        config=None,
        valid_num=0,  # take as valid aspect-ratio when sample number >= valid_num
        hq_only=False,
        cache_file=None,
        caching=False,
        clipscore_filter_thres=0.0,
        **kwargs,
    ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError(f"sampler should be an instance of ``Sampler``, but got {sampler}")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")

        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios
        self.drop_last = drop_last
        self.hq_only = hq_only
        self.config = config
        self.caching = caching
        self.cache_file = cache_file
        self.order_check_pass = False
        self.clipscore_filter_thres = clipscore_filter_thres

        self.ratio_nums_gt = kwargs.get("ratio_nums", None)
        assert self.ratio_nums_gt, "ratio_nums_gt must be provided."
        self._aspect_ratio_buckets = {ratio: [] for ratio in aspect_ratios.keys()}
        self.current_available_bucket_keys = [str(k) for k, v in self.ratio_nums_gt.items() if v >= valid_num]

        logger = (
            get_root_logger()
            if (config is None or config.work_dir is None)
            else get_root_logger(os.path.join(config.work_dir, "train_log.log"))
        )
        logger.warning(
            f"Image sampler using valid_num={valid_num} in config file. Available {len(self.current_available_bucket_keys)} aspect_ratios: {self.current_available_bucket_keys}"
        )

        self.data_all = {} if caching else None
        if cache_file is not None and os.path.exists(cache_file):
            logger.info(f"Loading cached file for multi-scale training: {cache_file}")
            try:
                self.cached_idx = json.load(open(cache_file))
            except:
                logger.info(f"Failed loading: {cache_file}")
                self.cached_idx = {}
        else:
            logger.info(f"No cached file is found, dataloader is slow: {cache_file}")
            self.cached_idx = {}
        self.exist_ids = len(self.cached_idx)

    def __iter__(self) -> Sequence[int]:
        for idx in self.sampler:
            data_info, closest_ratio = self._get_data_info_and_ratio(idx)
            if not data_info:
                continue

            bucket = self._aspect_ratio_buckets[closest_ratio]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                self._update_cache(bucket)
                yield bucket[:]
                del bucket[:]

        for bucket in self._aspect_ratio_buckets.values():
            while bucket:
                if not self.drop_last or len(bucket) == self.batch_size:
                    yield bucket[:]
                del bucket[:]

    def _get_data_info_and_ratio(self, idx):
        str_idx = str(idx)
        if self.caching:
            if str_idx in self.cached_idx:
                return self.cached_idx[str_idx], self.cached_idx[str_idx]["closest_ratio"]
            data_info = self.dataset.get_data_info(int(idx))
            if data_info is None or (
                self.hq_only and "version" in data_info and data_info["version"] not in ["high_quality"]
            ):
                return None, None
            closest_ratio = self._get_closest_ratio(data_info["height"], data_info["width"])
            self.data_all[str_idx] = {
                "height": data_info["height"],
                "width": data_info["width"],
                "closest_ratio": closest_ratio,
                "key": data_info["key"],
            }
            return data_info, closest_ratio
        else:
            if self.cached_idx:
                if self.cached_idx.get(str_idx):
                    if not self.order_check_pass or random.random() < 0.01:
                        # Ensure the cached dataset is in the same order as the original tar file
                        self._order_check(str_idx)
                    closest_ratio = self.cached_idx[str_idx]["closest_ratio"]
                    return self.cached_idx[str_idx], closest_ratio

            data_info = self.dataset.get_data_info(int(idx))
            if (
                data_info is None
                or (self.hq_only and "version" in data_info and data_info["version"] not in ["high_quality"])
                or (data_info.get("clipscore", False) and data_info["clipscore"] < self.clipscore_filter_thres)
            ):
                return None, None

            closest_ratio = self._get_closest_ratio(data_info["height"], data_info["width"])

            return data_info, closest_ratio

    def _get_closest_ratio(self, height, width):
        ratio = height / width
        return min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))

    def _order_check(self, str_idx):
        ori_data = self.cached_idx[str_idx]
        real_key = self.dataset.get_data_info(int(str_idx))["key"]
        assert real_key and ori_data["key"] == real_key, ValueError(
            f"index: {str_idx}, real key: {real_key} ori key: {ori_data['key']}"
        )
        self.order_check_pass = True

    def _update_cache(self, bucket):
        if self.caching:
            for idx in bucket:
                if str(idx) in self.cached_idx:
                    continue
                self.cached_idx[str(idx)] = self.data_all.pop(str(idx))


class AspectRatioBatchSamplerVideo(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(
        self,
        sampler: Sampler,
        dataset: Dataset,
        batch_size: int,
        aspect_ratios: dict,
        drop_last: bool = False,
        config=None,
        valid_num=0,  # take as valid aspect-ratio when sample number >= valid_num
        hq_only=False,
        caching=False,
        clipscore_filter_thres=0.0,
        **kwargs,
    ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError(f"sampler should be an instance of ``Sampler``, but got {sampler}")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")

        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios
        self.drop_last = drop_last
        self.hq_only = hq_only
        self.config = config
        self.caching = caching
        self.order_check_pass = False
        self.clipscore_filter_thres = clipscore_filter_thres

        self.ratio_nums_gt = kwargs.get("ratio_nums", None)
        assert self.ratio_nums_gt, "ratio_nums_gt must be provided."
        self._aspect_ratio_buckets = {float(ratio): [] for ratio in aspect_ratios.keys()}
        self.current_available_bucket_keys = [str(k) for k, v in self.ratio_nums_gt.items() if v >= valid_num]

        logger = (
            get_root_logger()
            if (config is None or config.work_dir is None)
            else get_root_logger(os.path.join(config.work_dir, "train_log.log"))
        )
        logger.warning(
            f"Video sampler using valid_num={valid_num} in config file. Available {len(self.current_available_bucket_keys)} aspect_ratios: {self.current_available_bucket_keys}"
        )

    def __iter__(self) -> Sequence[int]:
        for idx in self.sampler:
            data_info, closest_ratio = self._get_data_info_and_ratio(idx)
            if not data_info:
                continue

            bucket = self._aspect_ratio_buckets[closest_ratio]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

        for bucket in self._aspect_ratio_buckets.values():
            while bucket:
                if not self.drop_last or len(bucket) == self.batch_size:
                    yield bucket[:]
                del bucket[:]

    def _get_data_info_and_ratio(self, idx):
        str_idx = str(idx)
        data_info = self.dataset.get_data_info(int(idx))

        if data_info is None:
            return None, None

        if "closest_ratio" in data_info:
            closest_ratio = data_info["closest_ratio"]
        else:
            closest_ratio = self._get_closest_ratio(data_info["height"], data_info["width"])

        closest_ratio = float(closest_ratio)

        return data_info, closest_ratio

    def _get_closest_ratio(self, height, width):
        ratio = float(height) / float(width)
        return min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))


class BalancedAspectRatioBatchSampler(AspectRatioBatchSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Assign samples to each bucket
        self.ratio_nums_gt = kwargs.get("ratio_nums", None)
        assert self.ratio_nums_gt
        self._aspect_ratio_buckets = {float(ratio): [] for ratio in self.aspect_ratios.keys()}
        self.original_buckets = {}
        self.current_available_bucket_keys = [k for k, v in self.ratio_nums_gt.items() if v >= 3000]
        self.all_available_keys = deepcopy(self.current_available_bucket_keys)
        self.exhausted_bucket_keys = []
        self.total_batches = len(self.sampler) // self.batch_size
        self._aspect_ratio_count = {}
        for k in self.all_available_keys:
            self._aspect_ratio_count[float(k)] = 0
            self.original_buckets[float(k)] = []
        logger = get_root_logger(os.path.join(self.config.work_dir, "train_log.log"))
        logger.warning(
            f"Available {len(self.current_available_bucket_keys)} aspect_ratios: {self.current_available_bucket_keys}"
        )

    def __iter__(self) -> Sequence[int]:
        i = 0
        for idx in self.sampler:
            data_info = self.dataset.get_data_info(idx)
            height, width = data_info["height"], data_info["width"]
            ratio = height / width
            closest_ratio = float(min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio)))
            if closest_ratio not in self.all_available_keys:
                continue
            if self._aspect_ratio_count[closest_ratio] < self.ratio_nums_gt[closest_ratio]:
                self._aspect_ratio_count[closest_ratio] += 1
                self._aspect_ratio_buckets[closest_ratio].append(idx)
                self.original_buckets[closest_ratio].append(idx)  # Save the original samples for each bucket
            if not self.current_available_bucket_keys:
                self.current_available_bucket_keys, self.exhausted_bucket_keys = self.exhausted_bucket_keys, []

            if closest_ratio not in self.current_available_bucket_keys:
                continue
            key = closest_ratio
            bucket = self._aspect_ratio_buckets[key]
            if len(bucket) == self.batch_size:
                yield bucket[: self.batch_size]
                del bucket[: self.batch_size]
                i += 1
                self.exhausted_bucket_keys.append(key)
                self.current_available_bucket_keys.remove(key)

        for _ in range(self.total_batches - i):
            key = choice(self.all_available_keys)
            bucket = self._aspect_ratio_buckets[key]
            if len(bucket) >= self.batch_size:
                yield bucket[: self.batch_size]
                del bucket[: self.batch_size]

                # If a bucket is exhausted
                if not bucket:
                    self._aspect_ratio_buckets[key] = deepcopy(self.original_buckets[key][:])
                    shuffle(self._aspect_ratio_buckets[key])
            else:
                self._aspect_ratio_buckets[key] = deepcopy(self.original_buckets[key][:])
                shuffle(self._aspect_ratio_buckets[key])
