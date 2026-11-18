"""The same per-image aspect-preserving budget reaches Qwen and H3 VAE."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from runtime import stage1
from runtime.qwen_ops import media

BUDGETS = ((672 * 384, "stage1_draft"), (1344 * 768, "final_output"))


class ReferenceImageMatchTests(unittest.TestCase):
    def test_each_budget_scales_each_aspect_independently(self):
        for (area, source), expected in zip(BUDGETS, (
                ((384, 672), (672, 384), (512, 512)),
                ((768, 1344), (1344, 768), (1024, 1024)))):
            with self.subTest(budget_source=source):
                actual = [media.resolve_reference_image_size(*size, target_area=area)
                          for size in ((2688, 1536), (1536, 2688), (2048, 2048))]
                self.assertEqual(actual, list(expected))

    def test_small_aligned_images_are_not_upscaled_and_old_route_remains(self):
        for area, source in BUDGETS:
            with self.subTest(budget_source=source):
                self.assertEqual(media.resolve_reference_image_size(320, 192, target_area=area), (192, 320))
                self.assertEqual(media.resolve_reference_image_size(192, 320, target_area=area), (320, 192))
        self.assertEqual(media.resolve_reference_image_size(1344, 768), (2048, 3584))
        self.assertIsNone(media.reference_target_area({"stage1": {}}))

    def test_same_config_budget_binds_native_resolver(self):
        for budget, source in BUDGETS:
            with self.subTest(budget_source=source):
                config = {"stage1": {"reference_image_resize": {
                    "mode": "match", "pixel_budget": budget, "budget_source": source}}}
                area = media.reference_target_area(config)
                self.assertEqual(area, budget)
                self.assertEqual(media.reference_target_area(config["stage1"]), budget)
                native = SimpleNamespace(resolve_reference_image_size=lambda *args: None)
                stage1.install_reference_image_size(native, area)
                for size in ((2688, 1536), (1536, 2688), (2048, 2048), (320, 192)):
                    self.assertEqual(native.resolve_reference_image_size(*size),
                                     media.resolve_reference_image_size(*size, target_area=area))
        with self.assertRaises(ValueError):
            media.reference_target_area({"reference_image_resize": {"mode": "stretch", "pixel_budget": 1}})

    def test_three_references_keep_order_aspect_and_pixels_without_crop(self):
        sizes = [(2688, 1536), (1536, 2688), (320, 192)]
        images = {str(index): Image.new("RGB", size, (index * 70, 31, 15))
                  for index, size in enumerate(sizes)}
        refs = [{"type": "image", "path": str(index)} for index in range(3)]
        for (budget, source), shapes in zip(BUDGETS, (
                [[384, 672, 3], [672, 384, 3], [192, 320, 3]],
                [[768, 1344, 3], [1344, 768, 3], [192, 320, 3]])):
            with self.subTest(budget_source=source), \
                    patch.object(media, "load_rgb", side_effect=images.__getitem__):
                kwargs, facts = media.prepare_inputs({"task": "ref2va", "references": refs},
                    SimpleNamespace(from_numpy=lambda value: value), target_area=budget)
                items = kwargs["minimax_ref_items"]
                self.assertEqual([fact["path"] for fact in facts], ["0", "1", "2"])
                self.assertEqual([fact["shape"] for fact in facts], shapes)
                for index, (item, fact) in enumerate(zip(items, facts)):
                    self.assertEqual(item["type"], "image")
                    expected = np.asarray(images[str(index)].resize((fact["shape"][1], fact["shape"][0]),
                                          Image.Resampling.LANCZOS)).astype(np.float32) / 255
                    np.testing.assert_array_equal(item["data"][0], expected)


if __name__ == "__main__":
    unittest.main()
