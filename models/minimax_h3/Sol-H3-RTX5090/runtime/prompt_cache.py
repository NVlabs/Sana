"""Load the exact offline post-connector AV context cache.

The actual official refiner consumes only video_encoding and audio_encoding.
The Gemma attention mask has already been consumed by process_hidden_states;
it is not a third input to the joint AV denoiser. This module imports no model
framework until explicitly extracting or loading a cache.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


FIXED_PROMPT = "4K, refined, high quality, cinematic detail, clean textures, natural motion."
RECIPE = {
    "boundary": "post_INT8_connector_video_and_audio_encoding",
    "text_encoder": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    "text_encoder_bytes": 15372969374,
    "connector": "ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors",
    "connector_bytes": 21504034224,
    "gemma_int8_linears": 328,
    "connector_int8_linears": 96,
    "weight_scale_dtype": "torch.float32",
    "context_dtype": "torch.bfloat16",
    "joint_audio_video": True,
    "mask_boundary": "consumed_inside_process_hidden_states_not_denoiser_input",
}
MAX_CACHE_BYTES = 64 * 1024**2
CONTEXT_SHAPES = {"video": (1, 1024, 4096), "audio": (1, 1024, 2048)}


def fingerprint(prompt, recipe=RECIPE):
    return hashlib.sha256(json.dumps({"prompt": prompt, "recipe": recipe},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_payload(payload, *, prompt, torch_module):
    """Bounded cache contract; no model weight scans or external downloads."""
    torch = torch_module
    if (not isinstance(prompt, str) or not prompt.strip() or not isinstance(payload, dict)
            or payload.get("schema_version") != 1 or payload.get("prompt") != prompt
            or payload.get("recipe") != RECIPE
            or payload.get("fingerprint") != fingerprint(prompt)):
        raise ValueError("fixed prompt cache prompt/model recipe mismatch")
    contexts, manifest = payload.get("contexts"), payload.get("tensor_manifest")
    if not isinstance(contexts, dict) or set(contexts) != {"video", "audio"} or not isinstance(manifest, dict):
        raise ValueError("fixed prompt cache requires both post-connector AV contexts")
    for name, tensor in contexts.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                or tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != CONTEXT_SHAPES[name]
                or any(size < 1 for size in tensor.shape)
                or tensor.numel() * tensor.element_size() > MAX_CACHE_BYTES
                or not bool(torch.isfinite(tensor).all())
                or manifest.get(name) != {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}):
            raise ValueError(f"invalid cached {name} context shape/dtype/finite manifest")
    stats = payload.get("prompt_stats")
    if (not isinstance(stats, dict) or stats.get("characters") != len(prompt)
            or stats.get("truncated") is not False):
        raise ValueError("fixed prompt cache requires original nontruncated tokenizer stats")
    return payload


class FixedPromptCache:
    def __init__(self, payload, *, prompt, torch_module, source_path=None):
        self.payload = validate_payload(payload, prompt=prompt, torch_module=torch_module)
        self.prompt, self.source_path = prompt, source_path
        self.context_calls = 0

    def require_prompt(self, prompt):
        if prompt != self.prompt:
            raise ValueError("case prompt cannot be used with fixed Stage2 conditioning")

    def stats(self, prompt):
        self.require_prompt(prompt)
        return {**self.payload["prompt_stats"], "source": "offline_fixed_prompt_cache"}

    def contexts(self, prompt, device):
        self.require_prompt(prompt)
        # Explicit clone also isolates callers on CPU, where .to() can alias.
        values = tuple(self.payload["contexts"][name].to(device=device).clone()
                       for name in ("video", "audio"))
        self.context_calls += 1
        return values

    def receipt(self):
        return {"prompt": self.prompt, "fingerprint": self.payload["fingerprint"],
            "source_path": self.source_path, "recipe": dict(RECIPE),
            "tensor_manifest": self.payload["tensor_manifest"],
            "stage2_gemma_loaded": False, "stage2_connector_loaded": False,
            "gemma_encode_calls": 0, "connector_process_calls": 0,
            "context_copy_calls": self.context_calls}


def load_cache(path, *, prompt, torch_module):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_CACHE_BYTES:
        raise ValueError("fixed prompt cache exceeds bounded feature-only file contract")
    payload = torch_module.load(path, map_location="cpu", weights_only=True)
    return FixedPromptCache(payload, prompt=prompt, torch_module=torch_module, source_path=str(path))
