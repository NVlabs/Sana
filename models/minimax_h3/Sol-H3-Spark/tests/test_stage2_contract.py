"""CPU-only handoff, normalization and offline-conditioning behavior checks."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import latent_transfer as transfer
from runtime import prompt_cache as cache
from runtime import stage2


class Tensor:
    def __init__(self, shape, expression="input", dtype="bf16", finite=True):
        self.shape, self.expression, self.dtype, self.finite = shape, expression, dtype, finite
        self.ndim, self.device, self.values = len(shape), "cuda:0", [1, 2]
    def to(self, *args, **kwargs):
        return self
    def clone(self):
        return copy.deepcopy(self)
    def numel(self):
        return math.prod(self.shape)
    def element_size(self):
        return 2
    def is_contiguous(self):
        return True
    def contiguous(self):
        return self
    def __sub__(self, other):
        return Tensor(self.shape, f"({self.expression}-{other.expression})")
    def __truediv__(self, other):
        return Tensor(self.shape, f"({self.expression}/{other.expression})")
    def __mul__(self, other):
        return Tensor(self.shape, f"({self.expression}*{other.expression})")
    def __add__(self, other):
        return Tensor(self.shape, f"({self.expression}+{other.expression})")
    def __getitem__(self, slices):
        return Tensor((*self.shape[:2], slices[2].stop, *self.shape[3:]), self.expression)


TORCH = SimpleNamespace(Tensor=Tensor, bfloat16="bf16",
    isfinite=lambda t: SimpleNamespace(all=lambda: t.finite))


def cache_payload():
    contexts = {name: Tensor(shape) for name, shape in cache.CONTEXT_SHAPES.items()}
    for tensor in contexts.values():
        tensor.device = SimpleNamespace(type="cpu")
    return {"schema_version": 1, "prompt": cache.FIXED_PROMPT, "recipe": dict(cache.RECIPE),
        "fingerprint": cache.fingerprint(cache.FIXED_PROMPT), "contexts": contexts,
        "tensor_manifest": {name: {"shape": list(t.shape), "dtype": str(t.dtype)} for name, t in contexts.items()},
        "prompt_stats": {"characters": len(cache.FIXED_PROMPT), "truncated": False}}


class Stage2ContractTests(unittest.TestCase):
    def test_cache_copies_both_modalities_without_mutating_template(self):
        cached = cache.FixedPromptCache(cache_payload(), prompt=cache.FIXED_PROMPT, torch_module=TORCH)
        first = cached.contexts(cache.FIXED_PROMPT, "cpu")
        first[0].values[0] = 99
        second = cached.contexts(cache.FIXED_PROMPT, "cpu")
        self.assertEqual(second[0].values, [1, 2])
        self.assertEqual(second[1].shape, (1, 1024, 2048))
        self.assertEqual(cached.context_calls, 2)
        with self.assertRaises(ValueError):
            cached.contexts("case-specific prompt", "cpu")

    def test_cache_rejects_precision_shape_recipe_and_missing_audio(self):
        mutations = [lambda p: p["contexts"].pop("audio"),
            lambda p: setattr(p["contexts"]["video"], "shape", (1, 1, 4096)),
            lambda p: setattr(p["contexts"]["audio"], "finite", False),
            lambda p: setattr(p["contexts"]["video"], "dtype", "fp32"),
            lambda p: p["recipe"].update(context_dtype="fp32"),
            lambda p: p["prompt_stats"].update(truncated=True)]
        for mutate in mutations:
            row = cache_payload()
            mutate(row)
            with self.assertRaises(ValueError):
                cache.validate_payload(row, prompt=cache.FIXED_PROMPT, torch_module=TORCH)

    def test_capture_requires_same_request_and_matching_payload_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = root / "stage1_direct_tensors.pt"
            payload.write_bytes(b"small tensor-only artifact")
            row = {"status": "PASS", "same_request": True, "external_anchor_used": False,
                "task": "t2va", "latent_only_transfer": True, "h3_decoder_calls": 0,
                "case_id": "case-a", "request_id": "warmup-a", "prompt": "a singer", "seed": 42,
                "source_index": 0, "payload_sha256": transfer.sha256(payload)}
            transfer.write_new(root / "capture.json", row)
            self.assertEqual(transfer.load_latent_capture(root, request_id="warmup-a")["case_id"], "case-a")
            with self.assertRaisesRegex(ValueError, "another request"):
                transfer.load_latent_capture(root, request_id="formal-a")
            payload.write_bytes(b"changed artifact")
            with self.assertRaisesRegex(ValueError, "SHA"):
                transfer.load_latent_capture(root, request_id="warmup-a")

    def test_prepare_is_model_free_and_active_request_cannot_be_replaced(self):
        value = object.__new__(stage2.Session)
        value.closed = value.failed = False
        value.prepared = value.current_capture = None
        with tempfile.TemporaryDirectory() as temp:
            result = value.prepare("request-a", Path(temp) / "stage2")
            self.assertFalse(result["stage2_gemma_loaded"])
            self.assertFalse(result["stage2_connector_loaded"])
            with self.assertRaisesRegex(RuntimeError, "idle"):
                value.prepare("request-b", Path(temp) / "other")
            with self.assertRaisesRegex(RuntimeError, "matching"):
                value.run("unused", Path(temp) / "stage2", "wrong-request")

    def test_upscaler_author_normalization_precedes_adapter_and_temporal_crop(self):
        calls = []
        def upscaler(value, **kwargs):
            calls.append(("upscaler", value.expression, kwargs))
            return Tensor(stage2.H3_UPSCALED, "model")
        upscaler.comfy_latents_mean = Tensor((1, 24, 1, 1, 1), "mean")
        upscaler.comfy_latents_std = Tensor((1, 24, 1, 1, 1), "std")
        def convert(value, **kwargs):
            calls.append(("adapter", value.expression, kwargs))
            return Tensor(stage2.ADAPTER_OUTPUT)
        base = SimpleNamespace(_timed_cuda=lambda fn: (fn(), 0.1))
        compat = SimpleNamespace(OfficialCompatRefiner=object)
        session = SimpleNamespace(torch=TORCH)
        cls = stage2._refiner_class(base, compat, session)
        model = object.__new__(cls)
        model.dtype, model.device = "bf16", "cuda:0"
        model.h3_upscaler, model.h3_ltx_adapter = upscaler, SimpleNamespace(convert=convert)
        model.h3_upscaler_calls = model.adapter_convert_calls = 0
        value = model.video_encode(Tensor(stage2.H3_INPUT))
        self.assertEqual(value.shape, stage2.REFINER_INPUT)
        self.assertEqual(calls[0][1], "((input-mean)/std)")
        self.assertEqual(calls[1][1], "((model*std)+mean)")
        self.assertEqual(calls[1][2], {"pixel_frames": 124, "pixel_height": 768,
            "pixel_width": 1344, "input_normalization": "normalized"})
        self.assertEqual(model.encode_highres_first_frame(None), [])


if __name__ == "__main__":
    unittest.main()
