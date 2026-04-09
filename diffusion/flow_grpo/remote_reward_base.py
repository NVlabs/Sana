import os
import time
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from diffusion.flow_grpo.reward_client import CosmosImageRewardClient


class RemoteImageRewardScorer:
    """Adapter to use remote image reward server in training-style scorer API.

    Subclasses should override at least:
      - ``build_request_prompts`` when prompt format differs
      - ``build_extra_data`` when request json needs extra fields
      - ``extract_scores`` when response key/shape differs
    """

    def __init__(
        self,
        reward_name: str,
        response_key: Optional[str] = None,
        host: Optional[str] = None,
        token: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None,
        poll_interval: float = 1.0,
        max_wait_seconds: float = 120.0,
        wait_before_fetch: Optional[float] = None,
    ):
        self.reward_name = reward_name
        self.response_key = response_key or reward_name
        self.extra_data = extra_data or {}
        self.poll_interval = float(poll_interval)
        self.max_wait_seconds = float(max_wait_seconds)
        # Match reward_client.py behavior: wait before first pull.
        default_wait = float(os.environ.get("REMOTE_REWARD_WAIT_BEFORE_FETCH", "20.0"))
        self.wait_before_fetch = float(default_wait if wait_before_fetch is None else wait_before_fetch)
        self.client = CosmosImageRewardClient(
            host=(host or os.environ.get("REMOTE_REWARD_HOST") or "https://cosmos-reward-image.dgxc-lepton.nvidia.com"),
            token=(token if token is not None else os.environ.get("REMOTE_REWARD_TOKEN")),
            reward_fn={reward_name: 1.0},
            name=reward_name,
        )

    def build_request_prompts(self, prompts: Sequence[str]) -> list[str]:
        """Build prompt list sent to remote server."""
        return list(prompts)

    def build_extra_data(self) -> Dict[str, Any]:
        """Build extra request fields sent to remote server."""
        return dict(self.extra_data)

    def extract_scores(self, result: Dict[str, Any], batch_size: int) -> np.ndarray:
        """Extract reward vector from remote response."""
        score_dict = result.get("scores", {})
        if self.response_key not in score_dict:
            raise RuntimeError(
                f"[{self.reward_name}] response missing expected key '{self.response_key}'. "
                f"Available keys: {list(score_dict.keys())}. Raw response: {result}"
            )
        return self._normalize_length(score_dict[self.response_key], batch_size)

    @staticmethod
    def _to_uint8_nhwc(images: Any) -> np.ndarray:
        if isinstance(images, torch.Tensor):
            # NCHW float [0,1] -> NHWC uint8
            arr = (images * 255.0).round().clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
            if arr.ndim != 4:
                raise ValueError(f"Expected 4D tensor images, got shape {arr.shape}")
            return arr.transpose(0, 2, 3, 1)
        arr = np.asarray(images)
        if arr.ndim != 4:
            raise ValueError(f"Expected 4D images, got shape {arr.shape}")
        # Accept NHWC uint8 directly.
        if arr.shape[-1] == 3:
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return arr
        # Accept NCHW.
        if arr.shape[1] == 3:
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return arr.transpose(0, 2, 3, 1)
        raise ValueError(f"Unsupported image shape {arr.shape}. Need NHWC or NCHW with 3 channels.")

    @staticmethod
    def _normalize_length(scores: Sequence[float], batch_size: int) -> np.ndarray:
        arr = np.asarray(scores, dtype=np.float32).reshape(-1)
        if arr.shape[0] == batch_size:
            return arr
        raise RuntimeError(f"Remote reward result length mismatch: got {arr.shape[0]}, expected {batch_size}.")

    def __call__(self, images: Any, prompts: Sequence[str]) -> np.ndarray:
        images_np = self._to_uint8_nhwc(images)
        prompts = self.build_request_prompts(prompts)
        batch_size = images_np.shape[0]
        if len(prompts) != batch_size:
            raise ValueError(f"Prompt count ({len(prompts)}) != image batch size ({batch_size})")

        try:
            task = self.client.submit_task(prompts, images_np, extra_data=self.build_extra_data())
        except Exception as e:
            raise RuntimeError(
                f"[{self.reward_name}] submit_task failed. "
                f"host={self.client.host}, token_set={self.client.token is not None}. "
                f"Original error: {e}"
            ) from e

        if self.wait_before_fetch > 0:
            time.sleep(self.wait_before_fetch)

        deadline = time.time() + self.max_wait_seconds
        reward_type = task["reward_types"][0]
        while time.time() < deadline:
            try:
                result = self.client.check_results(task["uuid"], reward_type, replica_id=task["replica_id"])
            except Exception as e:
                raise RuntimeError(
                    f"[{self.reward_name}] check_results failed for uuid={task['uuid']}, "
                    f"reward_type={reward_type}. Original error: {e}"
                ) from e

            if result is None:
                time.sleep(self.poll_interval)
                continue

            return self.extract_scores(result, batch_size)

        raise TimeoutError(
            f"[{self.reward_name}] timed out waiting for reward result after {self.max_wait_seconds}s. "
            f"uuid={task['uuid']}, reward_type={reward_type}"
        )
