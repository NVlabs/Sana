"""CPU task/conditioning contracts without importing GPU frameworks."""
from types import SimpleNamespace
import unittest

from runtime.stage1_ops.lookup import conditioning_lookup_plan
from runtime.stage1_ops.tasks import input_spec, ref_lora_alpha, task_of, validate_prepared_media


class MultimodalStage1Contracts(unittest.TestCase):
    def test_condition_lookup_preserves_native_gemm_row_counts(self):
        video, audio = [0.0, 0.1, 0.2, 0.3], [0.0, 0.4, 0.5, 0.6]
        text = conditioning_lookup_plan(video, audio)
        visual = conditioning_lookup_plan(video, audio, video_condition=True)
        av = conditioning_lookup_plan(video, audio, video_condition=True, audio_condition=True)
        self.assertEqual([len(row) for row in text], [1, 2, 2, 2])
        self.assertEqual([len(row) for row in visual], [2, 3, 3, 3])
        self.assertEqual([len(row) for row in av], [3, 4, 4, 4])
        self.assertEqual(visual[0], [0.0, 0.9990000128746033])
        self.assertEqual(av[0][-1], 1.0)
        with self.assertRaises(ValueError):
            conditioning_lookup_plan(video, audio, audio_condition=True)

    def test_ref_lora_cannot_silently_use_unit_scaling(self):
        self.assertEqual(ref_lora_alpha(128, None) / 128, 0.0625)
        self.assertEqual(ref_lora_alpha(128, 8), 8)
        for rank, alpha in ((8, None), (128, 128), (64, 8)):
            with self.assertRaises(RuntimeError):
                ref_lora_alpha(rank, alpha)

    def test_fl2va_retains_only_last_and_both_keyframe_order(self):
        for keys in ({"last_frame": "end.png"}, {"first_frame": "first.png", "last_frame": "end.png"}):
            case = {"task": "fl2va", **keys}
            media = [{"type": "image", "path": path, "shape": [384, 672, 3]}
                     for path in keys.values()]
            payload = {"input_spec": input_spec(case), "prepared_media": media}
            batch = SimpleNamespace(extra={"minimax_h3_keyframes": [
                SimpleNamespace(height=384, width=672) for _ in keys]}, references=None)
            validate_prepared_media(case, payload, batch)
            payload["prepared_media"][0]["shape"] = [768, 1344, 3]
            with self.assertRaises(RuntimeError):
                validate_prepared_media(case, payload, batch)

    def test_ref2va_keeps_order_and_embedded_video_audio(self):
        case = {"task": "ref2va", "references": [{"type": "video", "path": "scene.mp4"},
                                                   {"type": "audio", "path": "voice.wav"}]}
        payload = {"input_spec": input_spec(case), "prepared_media": [
            {"type": "video", "path": "scene.mp4", "shape": [124, 768, 1344, 3], "has_audio": True},
            {"type": "audio", "path": "voice.wav", "shape": None, "has_audio": True}]}
        batch = SimpleNamespace(references=[
            SimpleNamespace(media_type="video", frames=SimpleNamespace(shape=(124, 768, 1344, 3)), has_audio=True),
            SimpleNamespace(media_type="audio", has_audio=True)])
        validate_prepared_media(case, payload, batch)
        payload["prepared_media"][0]["has_audio"] = False
        with self.assertRaises(RuntimeError):
            validate_prepared_media(case, payload, batch)

    def test_t2va_still_forbids_condition_rows(self):
        self.assertEqual(task_of({}), "t2va")
        with self.assertRaises(ValueError):
            task_of({"task": "unknown"})
        with self.assertRaises(RuntimeError):
            validate_prepared_media({}, {}, SimpleNamespace(extra={"minimax_h3_keyframes": [object()]}, references=None))


if __name__ == "__main__":
    unittest.main()
