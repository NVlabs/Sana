"""CPU pixel-policy and native Comfy call/handoff tests; no GPU imports."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from runtime import qwen
from runtime.qwen_ops import media


class FakeComfyClip:
    """The runtime must pass visual inputs through, not build its own tokens."""
    def __init__(self, check):
        self.check = check
        self.tags = np.array([1, 0, 0, 1], dtype=np.int64)
        self.cond = np.zeros((1, 4, 5120), dtype=np.float32)
        self.tokens = object()

    def tokenize(self, prompt, **kwargs):
        self.check(prompt, kwargs)
        return self.tokens

    def encode_from_tokens(self, tokens, return_dict):
        assert tokens is self.tokens and return_dict is True
        return {"cond": self.cond, "minimax_token_tags": self.tags}


class QwenMediaContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.image = Image.fromarray(np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3))
        self.first = str(self.root / "first.png")
        self.last = str(self.root / "last.png")
        self.image.save(self.first)
        Image.new("RGB", (8, 4), (255, 31, 15)).save(self.last)
        self.torch = SimpleNamespace(from_numpy=lambda value: value)

    def tearDown(self):
        self.temporary.cleanup()

    def test_keyframe_pixels_match_native_stretch_then_center_crop(self):
        stretched = media.prepare_keyframe_image(self.image, 6, 6, True)
        cropped = media.prepare_keyframe_image(self.image, 6, 6, False)
        expected_stretch = self.image.resize((6, 6), Image.Resampling.LANCZOS)
        expected_crop = self.image.resize((12, 6), Image.Resampling.LANCZOS).crop((3, 0, 9, 6))
        np.testing.assert_array_equal(stretched, expected_stretch)
        np.testing.assert_array_equal(cropped, expected_crop)
        self.assertFalse(np.array_equal(stretched, cropped))

    def test_last_only_is_first_existing_keyframe_and_stretches(self):
        kwargs, facts = media.prepare_inputs({"task": "fl2va", "last_frame": self.first}, self.torch)
        expected = np.array(self.image.resize((672, 384), Image.Resampling.LANCZOS)).astype(np.float32) / 255
        np.testing.assert_array_equal(kwargs["images"][0][0], expected)
        self.assertEqual(facts, [{"type": "image", "path": self.first, "anchor": "last",
                                 "shape": [384, 672, 3], "has_audio": False}])

    def test_exif_orientation_is_applied_before_geometry(self):
        path = self.root / "rotated.png"
        exif = Image.Exif()
        exif[274] = 6
        self.image.save(path, exif=exif)
        image = media.load_rgb(path)
        np.testing.assert_array_equal(image, self.image.transpose(Image.Transpose.ROTATE_270))

    def test_reference_image_uses_native_2048_upscale_and_round32(self):
        self.assertEqual(media.resolve_reference_image_size(8, 4), (2048, 4096))
        self.assertEqual(media.resolve_reference_image_size(1000, 777), (2048, 2624))
        with self.assertRaises(ValueError):
            media.resolve_reference_image_size(5, 1)
        prepared = media.prepare_reference_image(self.image, 32, 64)
        np.testing.assert_array_equal(prepared, self.image.resize((64, 32), Image.Resampling.LANCZOS))

    def test_video_canvas_and_resample_match_native_rounding(self):
        self.assertEqual(media.resolve_canvas_size(672, 384), (768, 1344))
        self.assertEqual(media.resolve_canvas_size(384, 672), (1344, 768))
        frames = np.arange(5, dtype=np.uint8)[:, None, None, None] * np.ones((5, 1, 1, 3), dtype=np.uint8)
        np.testing.assert_array_equal(media.resample_reference_frames(frames, 12), np.repeat(frames, 2, axis=0))
        np.testing.assert_array_equal(media.resample_reference_frames(frames, 30), frames[[0, 1, 3, 4]])
        self.assertIs(media.resample_reference_frames(frames, 24), frames)
        with self.assertRaises(ValueError):
            media.resample_reference_frames(frames, 0)

    def test_video_prep_trims124_before_reference_canvas_resize(self):
        frames = np.arange(126, dtype=np.uint8)[:, None, None, None] * np.ones((126, 1, 1, 3), dtype=np.uint8)
        with patch.object(media, "resolve_canvas_size", return_value=(2, 2)) as canvas:
            prepared = media.prepare_reference_frames(frames)
        canvas.assert_called_once_with(1, 1)
        self.assertEqual(prepared.shape, (124, 2, 2, 3))
        np.testing.assert_array_equal(prepared[:, 0, 0, 0], np.arange(124))

    def test_video_samples_are_unpadded_per_frame_timestamps(self):
        indices, timestamps = media.reference_video_samples(124)
        self.assertEqual(indices, list(range(0, 124, 12)))
        self.assertEqual(timestamps, [index / 2 for index in range(11)])
        # Comfy, not the caller, duplicates the odd last frame and averages pairs.
        padded = timestamps + timestamps[-1:]
        self.assertEqual([(padded[i] + padded[i + 1]) / 2 for i in range(0, 12, 2)],
                         [0.25, 1.25, 2.25, 3.25, 4.25, 5.0])

    def test_video_decoder_native_rotation_and_audio_presence_without_audio_decode(self):
        pixels = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        frame = SimpleNamespace(rotation=90, to_ndarray=lambda format: pixels)
        stream = SimpleNamespace(average_rate=24)
        class Container:
            streams = SimpleNamespace(video=[stream], audio=[object()])
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def decode(self, selected):
                assert selected is stream
                return iter([frame])
        with patch.dict(sys.modules, {"av": SimpleNamespace(open=lambda path: Container())}):
            frames, fps, has_audio = media.decode_reference_video("reference.mp4")
        np.testing.assert_array_equal(frames[0], np.rot90(pixels, k=-1))
        self.assertEqual(fps, 24)
        self.assertTrue(has_audio)

    def _run_fake_session(self, case, check, *, target_area=None):
        session = qwen.Session.__new__(qwen.Session)
        session.closed = session.failed = False
        session.completed_requests = 0
        session.model_load_count = 1
        session.reference_target_area = target_area
        session.clip = FakeComfyClip(check)
        session.normalize = lambda encoded: SimpleNamespace(
            cond=encoded["cond"], minimax_token_tags=encoded["minimax_token_tags"])
        session.release_idle_cache = lambda: {"status": "PASS"}
        payloads = []
        def save(payload, stream):
            payloads.append(payload)
            stream.write(b"fake tensor payload")
        session.torch = SimpleNamespace(from_numpy=lambda value: value,
            inference_mode=nullcontext, save=save, cuda=SimpleNamespace(
                reset_peak_memory_stats=Mock(), synchronize=Mock(),
                max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0))
        output = self.root / f"conditioning-{target_area}"
        receipt = session.run(case, str(output))
        payload, = payloads
        self.assertIs(payload["text_token_tags"], session.clip.tags)
        self.assertIs(payload["prompt_embeds"], session.clip.cond)
        self.assertEqual(payload["text_token_tags"].tolist(), [1, 0, 0, 1])
        for key in ("task", "external_anchor_used", "input_conditioned", "input_spec", "prepared_media"):
            self.assertEqual(payload[key], receipt[key])
        self.assertEqual(json.loads((output / "conditioning.json").read_text()), receipt)
        return receipt

    def test_t2va_keeps_native_empty_images_call(self):
        case = {"case_id": "text", "prompt": "Original prompt.", "seed": 42}
        def check(prompt, kwargs):
            self.assertEqual(prompt, case["prompt"])
            self.assertEqual(kwargs, {"images": []})
        receipt = self._run_fake_session(case, check)
        self.assertEqual(receipt["task"], "t2va")
        self.assertFalse(receipt["input_conditioned"])
        self.assertFalse(receipt["external_anchor_used"])
        self.assertEqual(receipt["prepared_media"], [])

    def test_fl2va_native_call_keeps_first_last_order_and_visual_tags(self):
        case = {"case_id": "endpoints", "prompt": "Original prompt.", "seed": 42,
                "task": "fl2va", "first_frame": self.first, "last_frame": self.last}
        def check(prompt, kwargs):
            self.assertEqual(prompt, case["prompt"])
            self.assertEqual(set(kwargs), {"images"})
            first, last = kwargs["images"]
            self.assertEqual(first.shape, (1, 384, 672, 3))
            expected_first = np.asarray(self.image.resize((672, 384), Image.Resampling.LANCZOS)).astype(np.float32) / 255
            np.testing.assert_array_equal(first[0], expected_first)
            np.testing.assert_array_equal(last[0, 0, 0], np.array([255, 31, 15], dtype=np.float32) / 255)
        receipt = self._run_fake_session(case, check)
        self.assertTrue(receipt["input_conditioned"])
        self.assertFalse(receipt["external_anchor_used"])
        self.assertEqual([item["anchor"] for item in receipt["prepared_media"]], ["first", "last"])
        self.assertEqual(receipt["input_spec"], {"task": "fl2va", "first_frame": self.first,
                                              "last_frame": self.last, "references": []})

    def test_ref2va_native_call_keeps_mixed_order_and_embedded_audio_label(self):
        refs = [{"type": "audio", "path": "speech.wav"},
                {"type": "video", "path": "with-audio.mp4"},
                {"type": "image", "path": self.first},
                {"type": "video", "path": "silent.mp4"}]
        case = {"case_id": "references", "prompt": "Original prompt.", "seed": 42,
                "task": "ref2va", "references": refs}
        frames = np.arange(25, dtype=np.uint8)[:, None, None, None] * np.ones((25, 2, 2, 3), dtype=np.uint8)
        def check(prompt, kwargs):
            self.assertEqual(prompt, case["prompt"])
            self.assertEqual(set(kwargs), {"minimax_ref_items"})
            items = kwargs["minimax_ref_items"]
            self.assertEqual([item["type"] for item in items], ["audio", "audio", "video", "image", "video"])
            self.assertEqual(items[:2], [{"type": "audio"}, {"type": "audio"}])
            self.assertEqual(items[2]["timestamps"], [0.0, 0.5, 1.0])
            self.assertEqual(items[2]["data"].shape, (3, 2, 2, 3))
            np.testing.assert_array_equal(items[2]["data"][:, 0, 0, 0], np.array([0, 12, 24], dtype=np.float32) / 255)
            self.assertEqual(items[3]["data"].shape, (1, 2048, 4096, 3))
        with patch.object(media, "decode_reference_video", side_effect=[(frames, 24, True), (frames, 24, False)]), \
             patch.object(media, "prepare_reference_frames", side_effect=lambda value: value):
            receipt = self._run_fake_session(case, check)
        self.assertEqual(receipt["input_spec"], {"task": "ref2va", "first_frame": None,
                                              "last_frame": None, "references": refs})
        facts = receipt["prepared_media"]
        self.assertEqual([item["path"] for item in facts], [item["path"] for item in refs])
        self.assertEqual([item["has_audio"] for item in facts], [True, True, False, False])
        self.assertEqual(facts[1]["shape"], [25, 2, 2, 3])
        self.assertEqual(facts[1]["sampled_indices"], [0, 12, 24])
        self.assertTrue(receipt["input_conditioned"])

    def test_ref2va_session_reads_and_propagates_configured_image_budget(self):
        refs = [{"type": "image", "path": str(index)} for index in range(3)]
        images = {str(index): Image.new("RGB", size, (index * 70, 31, 15))
                  for index, size in enumerate(((1344, 768), (768, 1344), (1024, 1024)))}
        case = {"case_id": "three-images", "task": "ref2va", "prompt": "Original prompt.\n",
                "seed": 42, "references": refs}
        for budget, source, shapes in (
                (672 * 384, "stage1_draft", [[384, 672, 3], [672, 384, 3], [512, 512, 3]]),
                (1344 * 768, "final_output", [[768, 1344, 3], [1344, 768, 3], [1024, 1024, 3]])):
            with self.subTest(budget_source=source):
                config = {"stage1": {"reference_image_resize": {
                    "mode": "match", "pixel_budget": budget, "budget_source": source}}}
                session = qwen.Session.__new__(qwen.Session)
                # Policy is read before any checkpoint access or GPU imports.
                with self.assertRaises(KeyError):
                    session.__init__({}, str(self.root), config)
                self.assertEqual(session.reference_target_area, budget)
                def check(prompt, kwargs):
                    self.assertEqual(prompt, case["prompt"])
                    items = kwargs["minimax_ref_items"]
                    self.assertEqual([list(item["data"].shape[1:]) for item in items], shapes)
                    for index, item in enumerate(items):
                        np.testing.assert_array_equal(item["data"][0, 0, 0],
                            np.array([index * 70, 31, 15], dtype=np.float32) / 255)
                with patch.object(media, "load_rgb", side_effect=images.__getitem__):
                    receipt = self._run_fake_session(case, check, target_area=session.reference_target_area)
                self.assertEqual([item["path"] for item in receipt["prepared_media"]], ["0", "1", "2"])
                self.assertEqual([item["shape"] for item in receipt["prepared_media"]], shapes)


if __name__ == "__main__":
    unittest.main()
