import os
from pathlib import Path
import sys
import unittest
from unittest import mock

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME_ROOT))

from patches import minimax_h3_decoding as decoding  # noqa: E402


class _FakeDevice:
    def synchronize(self):
        pass

    def empty_cache(self):
        pass

    def memory_allocated(self):
        return 0

    def memory_reserved(self):
        return 0


class _FakeVAE:
    def __init__(self):
        self.events = []
        self.tile_calls = []

    def parameters(self):
        return []

    def to(self, *, dtype):
        self.events.append(("dtype", dtype))
        return self

    def disable_offload(self):
        self.events.append("resident")

    def enable_offload(self):
        self.events.append("layerwise")

    def _run_tile_tasks(
        self, tiles, tile_indices, forward_fn, stack_tiling, cls_agg=None
    ):
        self.tile_calls.append(list(tile_indices))
        return [forward_fn(tiles[index]) for index in tile_indices]


class MiniMaxH3FullVAETests(unittest.TestCase):
    def test_restores_layerwise_offload_after_failure(self):
        vae = _FakeVAE()
        with (
            mock.patch.dict(os.environ, {"H3_FULL_VAE_AFTER_DENOISE": "1"}),
            mock.patch.object(
                decoding, "is_layerwise_offloaded_module", return_value=True
            ),
            mock.patch.object(decoding.torch, "get_device_module", _FakeDevice),
        ):
            with self.assertRaisesRegex(RuntimeError, "decode failed"):
                with decoding._full_video_vae_residency(vae):
                    self.assertEqual(vae.events, ["resident"])
                    raise RuntimeError("decode failed")

        self.assertEqual(vae.events, ["resident", "layerwise"])

    def test_disabled_path_is_noop(self):
        vae = _FakeVAE()
        with (
            mock.patch.dict(os.environ, {"H3_FULL_VAE_AFTER_DENOISE": "0"}),
            mock.patch.object(
                decoding, "is_layerwise_offloaded_module", return_value=True
            ),
        ):
            with decoding._full_video_vae_residency(vae):
                pass

        self.assertEqual(vae.events, [])

    def test_resident_dtype_is_applied_before_and_after_materialization(self):
        vae = _FakeVAE()
        with (
            mock.patch.dict(
                os.environ,
                {
                    "H3_FULL_VAE_AFTER_DENOISE": "1",
                    "H3_FULL_VAE_DTYPE": "bf16",
                },
            ),
            mock.patch.object(
                decoding, "is_layerwise_offloaded_module", return_value=True
            ),
            mock.patch.object(decoding.torch, "get_device_module", _FakeDevice),
        ):
            with decoding._full_video_vae_residency(vae):
                pass

        self.assertEqual(
            vae.events,
            [
                ("dtype", decoding.torch.bfloat16),
                "resident",
                ("dtype", decoding.torch.bfloat16),
                "layerwise",
            ],
        )

    def test_tile_batching_preserves_order(self):
        vae = _FakeVAE()
        tiles = list(range(8))
        with mock.patch.dict(
            os.environ, {"H3_FULL_VAE_TILE_BATCH_SIZE": "3"}
        ):
            with decoding._bounded_video_vae_tile_batch(vae):
                outputs = vae._run_tile_tasks(
                    tiles, list(range(8)), lambda value: value * 2, True
                )

        self.assertEqual(outputs, [value * 2 for value in tiles])
        self.assertEqual(vae.tile_calls, [[0, 1, 2], [3, 4, 5], [6, 7]])


if __name__ == "__main__":
    unittest.main()
