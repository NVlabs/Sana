"""Regression tests for adaptive projected guidance rescaling."""

import torch

from diffusion.guiders.adaptive_projected_guidance import rescale_noise_cfg


def test_rescale_constant_prediction_remains_finite_and_unchanged():
    """A zero-variance guided sample must not turn the denoising state into NaN."""
    noise_cfg = torch.stack([
        torch.zeros(2, 2, 2),
        torch.arange(8, dtype=torch.float32).reshape(2, 2, 2),
    ]).requires_grad_()
    noise_pred_text = torch.randn_like(noise_cfg)

    result = rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.5)
    result.sum().backward()

    assert torch.isfinite(result).all()
    assert torch.equal(result[0], noise_cfg.detach()[0])
    assert noise_cfg.grad is not None
    assert torch.isfinite(noise_cfg.grad).all()


def test_rescale_nonconstant_prediction_preserves_original_ratio():
    """Non-degenerate samples retain the standard standard-deviation ratio."""
    noise_cfg = torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2)
    noise_pred_text = torch.flip(noise_cfg, dims=[-1])
    factor = 0.35

    result = rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=factor)
    dims = list(range(1, noise_cfg.ndim))
    expected = noise_cfg * noise_pred_text.std(dims, keepdim=True) / noise_cfg.std(dims, keepdim=True)
    expected = factor * expected + (1 - factor) * noise_cfg

    assert torch.allclose(result, expected)
