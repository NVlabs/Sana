"""Small CPU tests for pipeline contracts; GPU validation is recorded separately."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from sol_refiner_h3 import DEFAULT_SIGMA, SoLRefinerH3Pipeline, output_geometry


class ConstantVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter("anchor", torch.nn.Parameter(torch.zeros(())))
        self.config = SimpleNamespace(
            in_channels=4, audio_in_channels=2, audio_cross_attention_dim=8
        )
        self.calls = []

    @property
    def device(self):
        return self.anchor.device

    @property
    def dtype(self):
        return self.anchor.dtype

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return torch.full_like(kwargs["hidden_states"], 0.25), torch.zeros_like(
            kwargs["audio_hidden_states"]
        )


class ContractTests(unittest.TestCase):
    def pipeline(self, shift=1):
        model = ConstantVelocity()
        pipe = SoLRefinerH3Pipeline(
            vae=None,
            transformer=model,
            latent_upsampler=None,
            diffusion_decoder=None,
            scheduler=FlowMatchEulerDiscreteScheduler(
                shift=shift, use_dynamic_shifting=False
            ),
        )
        return pipe, model

    def test_h3_geometry(self):
        self.assertEqual(output_geometry(1920, 1080, 124), (1920, 1088, 121))
        self.assertEqual(output_geometry(1280, 720, 121), (1280, 768, 121))
        self.assertEqual(output_geometry(64, 64, 1), (64, 64, 1))
        for args in [(0, 1080, 124), (1920, -1, 124), (1920, 1080, 0)]:
            with self.assertRaises(ValueError):
                output_geometry(*args)

    def test_exact_one_step_and_scaled_prompt_timestep(self):
        pipe, model = self.pipeline()
        x = torch.linspace(-1, 1, 120).reshape(1, 4, 2, 3, 5)
        pred = pipe.denoise_latents(x, torch.zeros(1, 2, 8), sigma=DEFAULT_SIGMA)
        torch.testing.assert_close(pred, x - DEFAULT_SIGMA * 0.25)
        self.assertEqual(len(model.calls), 1)
        call = model.calls[0]
        self.assertTrue(call["isolate_modalities"])
        self.assertEqual(call["num_frames"], 2)
        torch.testing.assert_close(call["sigma"], torch.tensor([DEFAULT_SIGMA * 1000]))
        torch.testing.assert_close(
            call["timestep"], torch.full((1, 30), DEFAULT_SIGMA * 1000)
        )
        self.assertEqual(tuple(call["audio_hidden_states"].shape), (1, 1, 2))
        self.assertEqual(len(pipe.scheduler.timesteps), 1)
        self.assertEqual(float(pipe.scheduler.sigmas[-1]), 0.0)
        pred2 = pipe.denoise_latents(x, torch.zeros(1, 2, 8), sigma=DEFAULT_SIGMA)
        torch.testing.assert_close(pred2, pred)

    def test_invalid_schedule_rejected(self):
        pipe, _ = self.pipeline(shift=2)
        with self.assertRaises(ValueError):
            pipe.denoise_latents(torch.zeros(1, 4, 2, 3, 5), torch.zeros(1, 2, 8))

    def test_invalid_model_input_rejected(self):
        pipe, _ = self.pipeline()
        for sigma in [0, -0.1, 1.1]:
            with self.assertRaises(ValueError):
                pipe.denoise_latents(
                    torch.zeros(1, 4, 2, 3, 5), torch.zeros(1, 2, 8), sigma=sigma
                )
        with self.assertRaises(ValueError):
            pipe.denoise_latents(torch.zeros(1, 8, 2, 3, 5), torch.zeros(1, 2, 8))


if __name__ == "__main__":
    unittest.main()
