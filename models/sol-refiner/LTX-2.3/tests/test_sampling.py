from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from sol_refiner import SoLRefinerPipeline
from sol_refiner.sampling import StepCache, sigma_schedule


class SamplingTests(unittest.TestCase):
    def pipeline(self, variant):
        return SoLRefinerPipeline(
            vae=None,
            transformer=None,
            latent_upsampler=None,
            scheduler=FlowMatchEulerDiscreteScheduler(),
            variant=variant,
        )

    def test_exact_schedules(self):
        torch.testing.assert_close(sigma_schedule("one-step"), torch.tensor([0.725, 0]))
        s = sigma_schedule("multi-step")
        self.assertEqual(len(s) - 1, 19)
        self.assertAlmostEqual(float(s[0]), 0.909375, places=6)
        self.assertEqual(float(s[-1]), 0)
        self.assertTrue(torch.all(s[:-1] > s[1:]))

    def test_steps_and_cfg_calls(self):
        x = torch.ones(1, 4, 2, 2, 2)
        pos, neg = torch.ones(1, 1, 8), torch.zeros(1, 1, 8)
        for variant, expected in [("one-step", 1), ("multi-step", 38)]:
            pipe = self.pipeline(variant)

            def velocity(x, context, *args):
                return torch.full_like(x, float(context.mean()) * 0.25)

            with patch.object(pipe, "_velocity", side_effect=velocity) as forward:
                out = pipe.denoise_latents(x, pos, negative_prompt_embeds=neg)
            self.assertEqual(forward.call_count, expected)
            scale = 1 if variant == "one-step" else 3
            torch.testing.assert_close(
                out, x - float(sigma_schedule(variant)[0]) * 0.25 * scale
            )
            with patch.object(pipe, "_velocity", side_effect=velocity):
                torch.testing.assert_close(
                    pipe.denoise_latents(x, pos, negative_prompt_embeds=neg), out
                )

    def test_cache_guards(self):
        pipe = self.pipeline("one-step")
        with self.assertRaises(ValueError):
            pipe.denoise_latents(
                torch.ones(1, 4, 2, 2, 2), torch.ones(1, 1, 8), teacache=True
            )
        cache = StepCache(threshold=0.15, start=3, max_hits=2)
        x = torch.ones(1, 4)
        for i in range(3):
            self.assertFalse(cache.reuse(x, i))
            cache.prediction = x
        self.assertTrue(cache.reuse(x, 3))
        self.assertTrue(cache.reuse(x, 4))
        self.assertFalse(cache.reuse(x, 5))
        self.assertFalse(cache.reuse(x * 2, 6))

    def test_cache_keeps_scheduler_updates(self):
        pipe = self.pipeline("multi-step")
        x, p = torch.ones(1, 4, 2, 2, 2), torch.ones(1, 1, 8)
        with (
            patch.object(
                pipe, "_velocity", side_effect=lambda x, *a: torch.zeros_like(x)
            ),
            patch.object(pipe.scheduler, "step", wraps=pipe.scheduler.step) as step,
        ):
            out = pipe.denoise_latents(x, p, negative_prompt_embeds=p, teacache=True)
        self.assertEqual(step.call_count, 19)
        self.assertGreater(pipe.last_run["cached_steps"], 0)
        self.assertLess(pipe.last_run["transformer_calls"], 38)
        torch.testing.assert_close(out, x)


if __name__ == "__main__":
    unittest.main()
