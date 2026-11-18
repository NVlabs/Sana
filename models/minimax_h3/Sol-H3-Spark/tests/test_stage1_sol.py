"""CPU-only checks of the fixed Ref2VA routing policy."""
from collections import Counter
from types import SimpleNamespace
import unittest

from runtime.stage1_ops.sol import RequestRoute, route, sink_plan, validate_request


class CpuTensor(list):
    device = SimpleNamespace(type="cpu")
    def tolist(self):
        return list(self)


class SolRouteTests(unittest.TestCase):
    def layout(self, text=86, ref_audio=0):
        # Native input order: Qwen (vision + true text), reference image/audio,
        # generated audio, then generated video. Counts match the two fixtures.
        # Actual native tags include two additional visual-tagged Qwen markers.
        tags = [0] * 7170 + [1] * text + [0] * 7168 + [2] * (ref_audio + 414) + [0] * 9324
        video = list(range(7170 + text, 14338 + text)) + list(range(len(tags) - 9324, len(tags)))
        return SimpleNamespace(token_tags=CpuTensor(tags), video_indices=CpuTensor(video),
                               num_condition_video_rows=7168)

    def test_exact_step_layer_schedule(self):
        counts = Counter(route(step, layer) for step in range(4) for layer in range(50))
        self.assertEqual(counts, {None: 53, 1.0: 49, 1.25: 49, 1.5: 49})
        for args in ((4, 0), (-1, 1), (1, 50)):
            with self.assertRaises(RuntimeError):
                route(*args)

    def test_actual_layouts_exclude_all_reference_image_and_qwen_vision_sinks(self):
        for text, audio, total, sinks in ((86, 0, 24162, 500), (108, 406, 24590, 928)):
            layout = self.layout(text, audio)
            permutation, inverse, receipt = sink_plan(layout.token_tags, layout.video_indices, 7168)
            self.assertEqual([permutation[inverse[i]] for i in range(total)], list(range(total)))
            self.assertEqual(receipt["sink_tokens"], sinks)
            self.assertEqual(receipt["sink_start"], 23662)
            self.assertEqual(receipt["boundary_generated_video_kv_tokens"], 46)
            self.assertEqual([i for i in permutation if layout.token_tags[i] == 0],
                             [i for i, tag in enumerate(layout.token_tags) if tag == 0])
            self.assertTrue(all(layout.token_tags[i] in (1, 2) for i in permutation[23662:]))

    def test_rejects_reference_image_boundary_spill(self):
        with self.assertRaises(RuntimeError):
            sink_plan([0] * 65 + [1, 2], list(range(65)), 65)

    def test_request_reset_runtime_routes_and_physical_counts(self):
        state = {}
        controller = RequestRoute(state)
        for _ in range(2):
            controller.prepare(self.layout())
            for step in range(4):
                controller.start_forward()
                for layer in range(50):
                    controller.record_body(layer, route(step, layer))
            receipt = state["sol_attention"]
            receipt.update(sol_calls=147, full_dense_fa4_calls=55, sink_query_fa4_calls=147,
                           tau_calls={"1.0": 49, "1.25": 49, "1.5": 49})
            self.assertEqual(validate_request(receipt)["physical_fa4_calls"], 202)
            with self.assertRaises(RuntimeError):
                controller.start_forward()
            receipt["sink_query_fa4_calls"] -= 1
            with self.assertRaises(RuntimeError):
                validate_request(receipt)


if __name__ == "__main__":
    unittest.main()
