"""CPU checks for fixed scheduling, VSA evidence and owned Qwen cleanup."""
import copy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from runtime import qwen, stage1
from runtime.stage1_ops.lookup import require_lookup_step, t2va_lookup_plan
from runtime.stage1_ops.vsa import COUNTERS, metadata_contract, validate_request
from runtime.qwen_ops.direct_io import DirectLoadError, iter_direct_chunks, preadv_fill_chunk


def vsa_receipt(requests):
    reference = 200 if requests else 0
    value = dict(zip(COUNTERS, (200 * requests, 200 * requests, 200 * requests,
                               0, 2 * requests, 200 * requests, reference)))
    value.update(active_compression_gates=50, selected_attention_backend="cudnn_bsa",
                 layer_calls=[4 * requests] * 50, step_calls=[50 * requests] * 4,
                 observed_metadata={}, warmup_selected_validation={
                     "status": "PASS", "passed_calls": reference},
                 bsa_index_order={"match_triton_order": True,
                                  "metadata_reorder_calls": 200 * requests},
                 bandwidth_optimization={"enabled": True,
                     "bshd_selected_calls": 200 * requests,
                     "fused_merge_calls": 200 * requests,
                     "warmup_layout_copies": 4 * reference})
    return value


class Stage1Contracts(unittest.TestCase):
    def test_native_prompt_normalization_preserves_internal_content(self):
        original = "\nsubject_definitions:\n<Subject 1> in <Picture 3>.\n\nsummary:\nDialogue.\n"
        stage1.validate_native_prompt(original.strip(), original)
        with self.assertRaises(RuntimeError):
            stage1.validate_native_prompt(original.strip().replace("Picture 3", "Picture 1"), original)
        with self.assertRaises(RuntimeError):
            stage1.validate_native_prompt(" ".join(original.split()), original)

    def test_exact_timestep_lookup_rejects_schedule_drift(self):
        plan = t2va_lookup_plan([0.0, 0.1, 0.2, 0.3], [0.0, 0.4, 0.5, 0.6])
        self.assertEqual([len(row) for row in plan], [1, 2, 2, 2])
        self.assertEqual(require_lookup_step(plan[2], plan, 6), 2)
        with self.assertRaises(RuntimeError):
            require_lookup_step([0.2, 0.7], plan, 6)
        with self.assertRaises(RuntimeError):
            t2va_lookup_plan([0.0] * 4, [0.0] * 4)

    def test_vsa_prefix_and_selected_workload(self):
        metadata = SimpleNamespace(VSA_sparsity=0.9, tile_elems=64, exempt=True,
            dense_layers=(), current_timestep=3, num_video_tiles=583, num_prefix_tiles=7)
        self.assertEqual(metadata_contract(metadata, 49)["selected_video_tiles_per_query"], 59)
        metadata.exempt = False
        with self.assertRaises(RuntimeError):
            metadata_contract(metadata, 49)

    def test_reference_runs_once_and_counters_cover_all_layers(self):
        first, second = vsa_receipt(1), vsa_receipt(2)
        self.assertEqual(validate_request(None, first)["request_counts"]["warmup_reference_calls"], 200)
        self.assertEqual(validate_request(first, second)["request_counts"]["warmup_reference_calls"], 0)
        missing = copy.deepcopy(second)
        missing["layer_calls"][13] -= 1
        with self.assertRaises(RuntimeError):
            validate_request(first, missing)

    def test_direct_io_accepts_eof_padding_but_rejects_missing_payload(self):
        chunk, = iter_direct_chunks(12, 10, alignment_bytes=16, chunk_bytes=32)
        storage = memoryview(bytearray(32))
        result = preadv_fill_chunk(0, storage, chunk, file_size=22, alignment_bytes=16,
                                   preadv_fn=lambda *args: 22)
        self.assertEqual(result, (22, 1, 0, 0, True))
        with self.assertRaises(DirectLoadError):
            preadv_fill_chunk(0, storage, chunk, file_size=21, alignment_bytes=16,
                              preadv_fn=lambda *args: 21)

    def test_qwen_close_unloads_only_owned_model(self):
        session = qwen.Session.__new__(qwen.Session)
        session.closed = False
        session._guards = []
        session.clip = SimpleNamespace(patcher=object())
        session.owner = object()
        session.torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=Mock()))
        owned = SimpleNamespace(model=session.clip.patcher, model_unload=Mock())
        unrelated = SimpleNamespace(model=object(), model_unload=Mock())
        management = SimpleNamespace(current_loaded_models=[owned, unrelated])
        comfy = SimpleNamespace(model_management=management)
        with patch.dict(sys.modules, {"comfy": comfy, "comfy.model_management": management}):
            session.close()
            session.close()
        owned.model_unload.assert_called_once_with()
        unrelated.model_unload.assert_not_called()
        self.assertEqual(management.current_loaded_models, [unrelated])


if __name__ == "__main__":
    unittest.main()
