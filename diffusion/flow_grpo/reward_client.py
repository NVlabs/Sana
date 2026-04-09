# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import requests
import torch


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


class CosmosImageRewardClient:
    """Client for the Cosmos image reward service.

    Supports HPSv2, ImageReward, OCR, and GenEval reward functions.
    Accepts raw RGB images (numpy uint8) instead of encoded latents.
    """

    def __init__(
        self,
        host: str = "http://localhost:8080",
        token: Optional[str] = None,
        reward_fn: Optional[Dict[str, float]] = None,
        name: str = "hpsv2",
    ):
        self.host = host.rstrip("/")
        self.token = token
        self.reward_fn = reward_fn or {"hpsv2": 1.0}
        self.name = name
        self.batch_size_dict: Dict[str, int] = {}

    def _make_headers(self, replica_id: Optional[str] = None, extra: Optional[dict] = None) -> dict:
        headers: dict = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if replica_id:
            headers["X-Lepton-Replica-Target"] = replica_id
        if extra:
            headers.update(extra)
        return headers

    def ping(self) -> dict:
        """Check server availability."""
        resp = requests.post(
            f"{self.host}/api/reward/ping",
            data={"info_data": json.dumps({"reward_fn": self.reward_fn})},
            headers=self._make_headers(),
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def submit_task(
        self,
        prompts: List[str],
        images: np.ndarray,
        reward_fn: Optional[Dict[str, float]] = None,
        extra_data: Optional[dict] = None,
    ) -> dict:
        """Submit images for reward calculation.

        Args:
            prompts: list of text prompts, one per image.
            images: uint8 numpy array of shape [B, H, W, 3].
            reward_fn: override reward function weights (defaults to self.reward_fn).
            extra_data: additional fields merged into the request JSON
                        (e.g. ocr_use_gpu, tag, include for GenEval).

        Returns:
            dict with "uuid", "replica_id", and reward type keys.
        """
        assert images.ndim == 4 and images.shape[-1] == 3, f"Expected images shape [B, H, W, 3], got {images.shape}"
        assert images.dtype == np.uint8, f"Expected uint8 images, got {images.dtype}"
        assert len(prompts) == images.shape[0], f"Prompt count ({len(prompts)}) != image count ({images.shape[0]})"

        self.batch_size_dict["uuid"] = images.shape[0]

        buf = io.BytesIO()
        np.save(buf, images, allow_pickle=False)
        npy_bytes = buf.getvalue()

        rw = reward_fn or self.reward_fn
        data = {
            "media_type": "image",
            "prompts": prompts,
            "reward_fn": rw,
        }
        if extra_data:
            data.update(extra_data)
        payload = json.dumps(data).encode("utf-8") + b"\n" + npy_bytes

        resp = requests.post(
            f"{self.host}/api/reward/enqueue",
            data=payload,
            headers=self._make_headers(extra={"Content-Type": "application/octet-stream"}),
            timeout=120,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Enqueue failed ({resp.status_code}): {resp.text}")

        result = resp.json()
        uuid = result["uuid"]
        replica_id = result.get("replica_id")
        reward_types = list(rw.keys())
        log(f"Task submitted. UUID: {uuid}, replica: {replica_id}")
        return {"uuid": uuid, "replica_id": replica_id, "reward_types": reward_types}

    def check_results(
        self,
        uuid: str,
        reward_type: str,
        replica_id: Optional[str] = None,
    ) -> Optional[dict]:
        """Single poll for results. Returns scores dict or None if still pending."""
        resp = requests.post(
            f"{self.host}/api/reward/pull",
            data={"uuid": uuid, "type": reward_type},
            headers=self._make_headers(replica_id=replica_id),
            timeout=20,
        )
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 400:
            return None
        else:
            raise RuntimeError(f"Pull failed ({resp.status_code}): {resp.text}")

    def check_results_until_get(
        self,
        uuid: str,
        reward_type: str,
        target_device: torch.device,
        replica_id: Optional[str] = None,
        poll_interval: float = 1.0,
    ) -> tuple:
        """Block until results are available, returns (processed_rewards, success_flag)."""
        while True:
            resp = requests.post(
                f"{self.host}/api/reward/pull",
                data={"uuid": uuid, "type": reward_type},
                headers=self._make_headers(replica_id=replica_id),
                timeout=20,
            )
            if resp.status_code == 200:
                return self.process_results(resp.json()["scores"], target_device), 1
            elif resp.status_code == 400:
                time.sleep(poll_interval)
            else:
                log(f"Pull error ({resp.status_code}): {resp.text}, returning zeros")
                return self.get_zero_rewards(self.batch_size_dict.get("uuid", 1), target_device), 0

    def get_zero_rewards(self, batch_size: int, target_device: torch.device) -> dict:
        weighted_rewards = torch.zeros(batch_size, dtype=torch.float32, device=target_device)
        return {
            self.name: weighted_rewards,
            "avg": [weighted_rewards[i] for i in range(batch_size)],
        }

    def process_results(self, scores: dict, target_device: torch.device) -> dict:
        rewards = {k: torch.as_tensor(v, device=target_device, dtype=torch.float32) for k, v in scores.items()}
        weighted = sum(rewards[k] * w for k, w in self.reward_fn.items() if k in rewards)
        if isinstance(weighted, int) and weighted == 0:
            batch_size = self.batch_size_dict.get("uuid", 1)
            weighted = torch.zeros(batch_size, dtype=torch.float32, device=target_device)
        return {
            self.name: weighted,
            "avg": [weighted[i] for i in range(weighted.shape[0])],
        }

    # ---- async wrappers (ThreadPoolExecutor) ----

    def submit_task_async(
        self,
        executor: ThreadPoolExecutor,
        prompts: List[str],
        images: np.ndarray,
        reward_fn: Optional[Dict[str, float]] = None,
        extra_data: Optional[dict] = None,
    ) -> Future:
        """Non-blocking submit. Returns a Future that resolves to the task dict."""
        return executor.submit(self.submit_task, prompts, images, reward_fn, extra_data)

    def fetch_results_async(
        self,
        executor: ThreadPoolExecutor,
        uuid: str,
        reward_type: str,
        target_device: torch.device,
        replica_id: Optional[str] = None,
        poll_interval: float = 1.0,
    ) -> Future:
        """Non-blocking fetch. Returns a Future that resolves to (rewards, flag)."""
        return executor.submit(
            self.check_results_until_get,
            uuid,
            reward_type,
            target_device,
            replica_id,
            poll_interval,
        )


# ---------- helpers ----------


def load_images_from_folder(folder_path: str) -> np.ndarray:
    """Load all images from a folder as a [B, H, W, 3] uint8 numpy array."""
    import os

    from PIL import Image

    imgs = []
    for name in sorted(os.listdir(folder_path)):
        path = os.path.join(folder_path, name)
        img = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        imgs.append(img)
    return np.stack(imgs, axis=0)


def make_fake_images(batch_size: int = 2, h: int = 512, w: int = 512) -> np.ndarray:
    """Generate random fake images for testing. Returns [B, H, W, 3] uint8."""
    return (np.random.rand(batch_size, h, w, 3) * 255.0).clip(0, 255).astype(np.uint8)


# ---------- main ----------


BATCH_SIZE = 8
WAIT_BEFORE_FETCH = 20.0


def _run_test(client: CosmosImageRewardClient, prompts, images, extra_data=None):
    """Submit a task, wait, then poll. Reports submit + fetch time (excluding wait)."""
    log(f"Images shape: {images.shape}, dtype: {images.dtype}")

    t0 = time.perf_counter()
    task = client.submit_task(prompts, images, extra_data=extra_data)
    submit_time = time.perf_counter() - t0
    log(f"UUID: {task['uuid']}, replica: {task['replica_id']}")
    log(f"Submit time: {submit_time:.3f}s")

    log(f"Waiting {WAIT_BEFORE_FETCH}s before fetching...")
    time.sleep(WAIT_BEFORE_FETCH)

    fetch_time = 0.0
    for rtype in task["reward_types"]:
        log(f"Polling results for reward type: {rtype} ...")
        while True:
            t1 = time.perf_counter()
            result = client.check_results(task["uuid"], rtype, replica_id=task["replica_id"])
            fetch_time += time.perf_counter() - t1
            if result is not None:
                log(f"Results ready: {result}")
                break
            time.sleep(1)

    log(f"Fetch time:  {fetch_time:.3f}s")
    log(f"Total time (submit + fetch, excluding wait): {submit_time + fetch_time:.3f}s")


def _simulate_gpu_work(duration: float, label: str = "GPU work"):
    """Simulate local GPU computation with a matmul workload."""
    log(f"[{label}] Starting (target ~{duration:.1f}s) ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration:
        a = torch.randn(1024, 1024, device=device)
        b = torch.randn(1024, 1024, device=device)
        _ = a @ b
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    log(f"[{label}] Done in {elapsed:.3f}s")
    return elapsed


def _run_async_demo(client: CosmosImageRewardClient, prompts, images, extra_data=None):
    """Demo: submit & fetch run in background threads while GPU keeps working."""
    executor = ThreadPoolExecutor(max_workers=2)
    gpu_work_time = 3.0

    log("=" * 60)
    log("SYNC baseline: submit -> wait -> fetch -> GPU work")
    log("=" * 60)
    t_sync_start = time.perf_counter()

    t0 = time.perf_counter()
    task = client.submit_task(prompts, images, extra_data=extra_data)
    sync_submit = time.perf_counter() - t0
    log(f"Submit: {sync_submit:.3f}s")

    time.sleep(WAIT_BEFORE_FETCH)

    t0 = time.perf_counter()
    for rtype in task["reward_types"]:
        while True:
            result = client.check_results(task["uuid"], rtype, replica_id=task["replica_id"])
            if result is not None:
                log(f"[{rtype}] got result")
                break
            time.sleep(1)
    sync_fetch = time.perf_counter() - t0
    log(f"Fetch: {sync_fetch:.3f}s")

    _simulate_gpu_work(gpu_work_time, "GPU work (sync)")

    sync_total = time.perf_counter() - t_sync_start
    log(f"SYNC total wall time: {sync_total:.3f}s")

    log("")
    log("=" * 60)
    log("ASYNC: submit overlaps GPU work, fetch overlaps GPU work")
    log("=" * 60)
    t_async_start = time.perf_counter()

    submit_future = client.submit_task_async(executor, prompts, images, extra_data=extra_data)
    _simulate_gpu_work(gpu_work_time, "GPU work during submit")

    task = submit_future.result()
    log(f"Submit finished (overlapped with GPU work)")

    time.sleep(WAIT_BEFORE_FETCH)

    fetch_future = client.fetch_results_async(
        executor,
        task["uuid"],
        task["reward_types"][0],
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        replica_id=task["replica_id"],
    )
    _simulate_gpu_work(gpu_work_time, "GPU work during fetch")

    rewards, flag = fetch_future.result()
    log(f"Fetch finished (overlapped with GPU work), success={flag}")

    async_total = time.perf_counter() - t_async_start
    log(f"ASYNC total wall time: {async_total:.3f}s")

    log("")
    log("=" * 60)
    log(f"SYNC  wall time: {sync_total:.3f}s")
    log(f"ASYNC wall time: {async_total:.3f}s")
    log(f"Saved: {sync_total - async_total:.3f}s ({(sync_total - async_total) / sync_total * 100:.1f}%)")
    log("=" * 60)

    executor.shutdown(wait=False)


if __name__ == "__main__":
    import argparse

    REWARD_CHOICES = ["hpsv2", "hpsv3", "pickscore", "image_reward", "ocr", "gen_eval", "async_demo"]
    # ocr pickscore hpsv2 hpsv3 image_reward

    parser = argparse.ArgumentParser(description="Test CosmosImageRewardClient")
    parser.add_argument(
        "reward_type",
        choices=REWARD_CHOICES,
        help=f"Reward type to test: {REWARD_CHOICES}",
    )
    parser.add_argument("--host", default="https://cosmos-reward-image.dgxc-lepton.nvidia.com")
    parser.add_argument("--token", default="YySuTH0PWzNIpDh4tWCXEvWSGYVCr8j4")
    args = parser.parse_args()

    rtype = args.reward_type
    log(f"=== Testing reward type: {rtype} ===")

    # ---- hpsv2 (+ image_reward combo) ----
    if rtype == "hpsv2":
        client = CosmosImageRewardClient(
            host=args.host,
            token=args.token,
            reward_fn={"hpsv2": 1.0, "image_reward": 1.0},
        )
        prompts = ["a photo of a cat", "a beautiful sunset"] * 4
        images = make_fake_images(batch_size=BATCH_SIZE)
        _run_test(client, prompts, images)

    # ---- hpsv3 ----
    elif rtype == "hpsv3":
        client = CosmosImageRewardClient(
            host=args.host,
            token=args.token,
            reward_fn={"hpsv3": 1.0},
        )
        prompts = [
            "cute chibi anime cartoon fox, smiling wagging tail " "with a small cartoon heart above sticker",
        ] * BATCH_SIZE
        images = make_fake_images(batch_size=BATCH_SIZE)
        _run_test(client, prompts, images)

    # ---- pickscore ----
    elif rtype == "pickscore":
        client = CosmosImageRewardClient(
            host=args.host,
            token=args.token,
            reward_fn={"pickscore": 1.0},
        )
        prompts = [
            "An astronaut's glove floating in zero-g " 'with "NASA 2049" on the wrist',
        ] * BATCH_SIZE
        images = make_fake_images(batch_size=BATCH_SIZE)
        _run_test(client, prompts, images)

    # ---- image_reward ----
    elif rtype == "image_reward":
        client = CosmosImageRewardClient(
            host=args.host,
            token=args.token,
            reward_fn={"image_reward": 1.0},
        )
        prompts = ["a photo of a cat", "a beautiful sunset"] * 4
        images = make_fake_images(batch_size=BATCH_SIZE)
        _run_test(client, prompts, images)

    # ---- ocr ----
    elif rtype == "ocr":
        client = CosmosImageRewardClient(
            host=args.host,
            token=args.token,
            reward_fn={"ocr": 1.0},
        )
        prompts = [
            'New York Skyline with "Hello World" written with fireworks',
        ] * BATCH_SIZE
        images = make_fake_images(batch_size=BATCH_SIZE)
        _run_test(client, prompts, images, extra_data={"ocr_use_gpu": True})

    # ---- gen_eval ----
    elif rtype == "gen_eval":
        client = CosmosImageRewardClient(
            host=args.host,
            token=args.token,
            reward_fn={"gen_eval": 1.0},
        )
        prompts = [
            "a photo of a brown giraffe and a white stop sign",
        ] * BATCH_SIZE
        images = make_fake_images(batch_size=BATCH_SIZE)
        _run_test(
            client,
            prompts,
            images,
            extra_data={
                "tag": "single_object",
                "include": [
                    {"class": "giraffe", "count": 1, "color": "brown"},
                    {"class": "stop sign", "count": 1, "color": "black"},
                ],
            },
        )

    # ---- async_demo: sync vs async comparison ----
    elif rtype == "async_demo":
        client = CosmosImageRewardClient(
            host=args.host,
            token=args.token,
            reward_fn={"hpsv2": 1.0},
        )
        prompts = ["a photo of a cat", "a beautiful sunset"] * 4
        images = make_fake_images(batch_size=BATCH_SIZE)
        _run_async_demo(client, prompts, images)
