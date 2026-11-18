"""Same-request normalized H3 latent and original PCM file handoff."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .config import TASKS

H3_SHAPE = (1, 24, 37, 24, 42)
AUDIO_SHAPE = (1, 2, 161333)
MAX_PAYLOAD_BYTES = 4 * 1024**2
WIRE_METADATA = {
    "input_variant": "h3_adapter", "primary_kind": "h3_normalized_latent",
    "normalization": "released_h3_per_channel_mean_std", "generation_frames": 124,
    "adapter_pixel_frames": 124, "padded_canvas_frames": 129, "emitted_frames": 121,
    "pixel_height": 384, "pixel_width": 672, "fps": 24.0, "audio_sample_rate": 32000,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def validate_request_id(request_id):
    if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", request_id):
        raise ValueError("invalid request_id")
    return request_id


def load_latent_capture(directory, *, request_id=None):
    root = Path(directory).resolve(strict=True)
    path = root / "capture.json"
    row = json.loads(path.read_text())
    if (row.get("status") != "PASS" or row.get("same_request") is not True
            or row.get("external_anchor_used") is not False or row.get("task") not in TASKS
            or row.get("latent_only_transfer") is not True
            or type(row.get("h3_decoder_calls")) is not int or row["h3_decoder_calls"] != 0
            or not isinstance(row.get("prompt"), str) or not row["prompt"].strip()
            or type(row.get("seed")) is not int or type(row.get("source_index")) is not int
            or row["source_index"] < 0):
        raise ValueError("invalid same-request latent-only H3 capture identity")
    validate_request_id(row.get("case_id"))
    if request_id is not None and row.get("request_id") != validate_request_id(request_id):
        raise ValueError("capture belongs to another request")
    payload = (root / "stage1_direct_tensors.pt").resolve(strict=True)
    payload.relative_to(root)
    if (not payload.is_file() or not 0 < payload.stat().st_size <= MAX_PAYLOAD_BYTES
            or sha256(payload) != row.get("payload_sha256")):
        raise ValueError("capture payload size or SHA mismatch")
    return {**row, "payload_path": str(payload), "capture_path": str(path),
            "capture_sha256": sha256(path)}


def load_payload(row, *, torch_module):
    torch = torch_module
    payload = torch.load(row["payload_path"], map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"h3_normalized", "audio"}:
        raise ValueError("unexpected Stage1 tensor keys")
    for key, shape, dtype in (("h3_normalized", H3_SHAPE, torch.bfloat16),
                              ("audio", AUDIO_SHAPE, torch.float32)):
        value = payload[key]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.dtype != dtype or not bool(torch.isfinite(value).all())):
            raise ValueError(f"invalid H3 {key} shape/dtype/finite contract")
        payload[key] = value.contiguous()
    return payload
