from __future__ import annotations

import torch

from diffusion.model.respace import process_timesteps


class _FakeTimeSampler:
    def __init__(self, num_chunks: int) -> None:
        self.num_chunks = num_chunks
        self.calls: list[dict[str, torch.Tensor | int | None]] = []

    def sample(self, batch_size: int, curf=None, cur_timestep=None) -> torch.Tensor:
        if cur_timestep is None:
            cur_timestep = torch.zeros(batch_size, dtype=torch.long)
        self.calls.append(
            {
                "batch_size": batch_size,
                "curf": None if curf is None else curf.detach().cpu().clone(),
                "cur_timestep": cur_timestep.detach().cpu().clone(),
            }
        )
        return cur_timestep[:, None].expand(batch_size, self.num_chunks).clone()


def _sample_with_mixture(mixture_probs: dict[str, float], sampler: _FakeTimeSampler) -> torch.Tensor:
    torch.manual_seed(123)
    return process_timesteps(
        weighting_scheme="logit_normal",
        train_sampling_steps=1000,
        size=(8, 1, 6),
        device=torch.device("cpu"),
        logit_mean=0.0,
        logit_std=1.0,
        num_frames=6,
        chunk_index=[0, 2, 4],
        chunk_sampling_strategy="incremental",
        chunk_mixture_probs=mixture_probs,
        time_sampler=sampler,
    )


def test_chunk_mixture_incremental_uses_time_sampler() -> None:
    sampler = _FakeTimeSampler(num_chunks=3)
    timesteps = _sample_with_mixture(
        {
            "same_t": 0.0,
            "incremental": 1.0,
            "last_chunk_anchor": 0.0,
            "teacher_forcing_clean": 0.0,
        },
        sampler,
    )

    assert timesteps.shape == (8, 1, 6)
    assert len(sampler.calls) == 1
    curf = sampler.calls[0]["curf"]
    assert isinstance(curf, torch.Tensor)
    assert curf.shape == (8,)
    assert int(curf.min()) >= 0
    assert int(curf.max()) < 3


def test_chunk_mixture_last_anchor_modes_force_last_chunk() -> None:
    sampler = _FakeTimeSampler(num_chunks=3)
    _sample_with_mixture(
        {
            "same_t": 0.0,
            "incremental": 0.0,
            "last_chunk_anchor": 1.0,
            "teacher_forcing_clean": 0.0,
        },
        sampler,
    )

    curf = sampler.calls[0]["curf"]
    assert isinstance(curf, torch.Tensor)
    assert torch.equal(curf, torch.full((8,), 2))


def test_chunk_mixture_teacher_forcing_cleans_prefix() -> None:
    sampler = _FakeTimeSampler(num_chunks=3)
    timesteps = _sample_with_mixture(
        {
            "same_t": 0.0,
            "incremental": 0.0,
            "last_chunk_anchor": 0.0,
            "teacher_forcing_clean": 1.0,
        },
        sampler,
    )

    curf = sampler.calls[0]["curf"]
    assert isinstance(curf, torch.Tensor)
    assert torch.equal(curf, torch.full((8,), 2))
    assert torch.equal(timesteps[:, :, :2], torch.zeros_like(timesteps[:, :, :2]))
