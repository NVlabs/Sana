"""Sampling contracts from the original LTX-2.3 refiner."""

import math

import torch


DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)


def sigma_schedule(variant):
    if variant == "one-step":
        return torch.tensor([0.725, 0.0])
    if variant != "multi-step":
        raise ValueError("variant must be one-step or multi-step")
    # The reference truncates a 32-step LTX schedule at sigma=0.909375.
    raw = torch.linspace(1.0, 0.0, 33)
    shifted = torch.where(raw != 0, math.exp(2.05) / (math.exp(2.05) + 1 / raw - 1), 0)
    shifted[:-1] = 1 - (1 - shifted[:-1]) / ((1 - shifted[-2]) / 0.9)
    sigmas = shifted[shifted <= 0.909375 + 1e-6].clone()
    sigmas[0] = 0.909375
    return sigmas


class StepCache:
    """Reference TeaCache policy: reuse the guided clean-latent prediction."""

    def __init__(self, threshold=0.15, start=3, max_hits=2):
        self.threshold, self.start, self.max_hits = threshold, start, max_hits
        self.previous = self.prediction = None
        self.accumulated = 0.0
        self.hits = self.skipped = 0

    def reuse(self, latent, step):
        current = latent.detach().float()
        reuse = False
        if (
            self.previous is not None
            and self.prediction is not None
            and step >= self.start
        ):
            self.accumulated += float(
                (current - self.previous).abs().mean()
                / (self.previous.abs().mean() + 1e-8)
            )
            reuse = self.accumulated < self.threshold and self.hits < self.max_hits
            if reuse:
                self.hits += 1
                self.skipped += 1
            else:
                self.accumulated = 0.0
                self.hits = 0
        self.previous = current
        return reuse
