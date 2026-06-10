from __future__ import annotations

import torch

from diffusion.scheduler.flow_euler_sampler import ChunkFlowEuler


class _RecordingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, object]] = []

    def forward(self, x, timestep, y, **kwargs):
        self.calls.append(
            {
                "frames": x.shape[2],
                "chunk_index": list(kwargs["chunk_index"]),
                "camera_frames": kwargs["camera_conditions"].shape[1],
                "plucker_frames": kwargs["chunk_plucker"].shape[2],
                "first_timestep": float(timestep[0, 0, 0].item()),
            }
        )
        return torch.zeros_like(x)


def test_chunk_flow_euler_uses_prefix_chunks_and_slices_conditions() -> None:
    model = _RecordingModel()
    latents = torch.randn(1, 1, 5, 1, 1)
    cond = torch.zeros(1, 1, 1)
    model_kwargs = {
        "data_info": {"condition_frame_info": {0: 0.0}},
        "camera_conditions": torch.randn(1, 5, 20),
        "chunk_plucker": torch.randn(1, 48, 5, 1, 1),
    }

    sampler = ChunkFlowEuler(
        model,
        condition=cond,
        uncondition=cond,
        cfg_scale=1.0,
        flow_shift=1.0,
        model_kwargs=model_kwargs,
    )
    out = sampler.sample(latents.clone(), steps=4, chunk_index=[0, 2, 4], interval_k=0.5)

    assert out.shape == latents.shape
    assert model.calls
    assert model.calls[0]["frames"] == 2
    assert model.calls[0]["chunk_index"] == [0]
    assert any(call["frames"] == 4 and call["chunk_index"] == [0, 2] for call in model.calls)
    assert model.calls[-1]["frames"] == 5
    assert model.calls[-1]["chunk_index"] == [0, 2, 4]
    for call in model.calls:
        assert call["camera_frames"] == call["frames"]
        assert call["plucker_frames"] == call["frames"]
        assert call["first_timestep"] == 0.0
