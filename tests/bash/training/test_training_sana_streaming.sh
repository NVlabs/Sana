#!/bin/bash
set -euo pipefail

TEST_DATA_DIR=data/sana_streaming_training_ci
TEST_OUTPUT_DIR=output/test_sana_streaming_training_ci

cleanup() {
    rm -rf "$TEST_DATA_DIR" "$TEST_OUTPUT_DIR"
}
trap cleanup EXIT

echo "Testing SANA-Streaming CPU training helpers"

python - <<'PY'
import copy
import io
import json
import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
import yaml

from diffusion.data import transforms as video_transforms
from diffusion.data.datasets import utils as dataset_utils
from diffusion.data.datasets.video import sana_v2v_pair_data
from diffusion.longsana.model.streaming_sana_long import StreamingSANATrainingModel
from diffusion.longsana.pipeline.sana_inference_pipeline import SanaInferencePipeline
from diffusion.longsana.pipeline.sana_reverse_reg_pipeline import SanaReverseRegPipeline
from diffusion.longsana.pipeline.sana_training_pipeline import SanaTrainingPipeline
from diffusion.longsana.trainer.longsana_trainer import LongSANATrainer
from diffusion.longsana.utils.dataset import LongV2VManifestDataset
from diffusion.scheduler.sana_streaming_sampler import SANAStreamingSampler


def expect_raises(exception_type, function, message):
    try:
        function()
    except exception_type as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"Expected {exception_type.__name__}: {message}")


def test_release_configs():
    bidirectional = yaml.safe_load(
        Path("configs/sana_streaming/train/sana_streaming_bidirectional_2b_720p.yaml").read_text(encoding="utf-8")
    )
    assert bidirectional["task"] == "v2v"
    assert bidirectional["data"]["data_dir"] == {
        "example_data": "data/sana_streaming_1k/data/example_data"
    }
    assert bidirectional["model"]["additional_inchannels"] == 128
    assert bidirectional["model"]["softmax_ratio"] == 0.25

    stage_441 = yaml.safe_load(
        Path("configs/sana_streaming/train/sana_streaming_long_441_2b_720p.yaml").read_text(encoding="utf-8")
    )
    stage_969 = yaml.safe_load(
        Path("configs/sana_streaming/train/sana_streaming_long_969_2b_720p.yaml").read_text(encoding="utf-8")
    )
    for config in (stage_441, stage_969):
        assert config["v2v"] is True
        assert config["trainer"] == "longsana"
        assert config["reverse_reg_weight"] == 0.5
        assert config["num_cached_blocks"] == 2
        assert config["sink_token"] is True
        assert "teacher" not in config
    assert stage_441["image_or_video_shape"][1] == 56
    assert stage_969["image_or_video_shape"][1] == 122
    assert stage_969["generator_ckpt"] == stage_969["fake_ckpt"]
    assert "sana_streaming_long_441_2b_720p" in stage_969["generator_ckpt"]


def test_long_v2v_manifest_dataset():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        videos = root / "videos"
        videos.mkdir()
        source_video = videos / "source.mp4"
        source_video.write_bytes(b"test video placeholder")

        manifest = root / "manifest.jsonl"
        manifest.write_text(
            "\n"
            + json.dumps(
                {
                    "prompt": "  apply an edit  ",
                    "reverse_prompt": "  restore the source  ",
                    "source_video": "videos/source.mp4",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        dataset = LongV2VManifestDataset(str(manifest))
        assert len(dataset) == 1
        assert dataset[0] == {
            "prompts": "apply an edit",
            "reverse_prompts": "restore the source",
            "source_video_paths": str(source_video.resolve()),
            "idx": 0,
        }

        escaped_manifest = root / "escaped.jsonl"
        escaped_manifest.write_text(
            json.dumps(
                {
                    "prompt": "apply an edit",
                    "reverse_prompt": "restore the source",
                    "source_video": "../outside.mp4",
                }
            ),
            encoding="utf-8",
        )
        expect_raises(
            ValueError,
            lambda: LongV2VManifestDataset(str(escaped_manifest)),
            "must stay within the local data root",
        )

        missing_field_manifest = root / "missing_field.jsonl"
        missing_field_manifest.write_text(
            json.dumps({"prompt": "apply an edit", "source_video": "videos/source.mp4"}),
            encoding="utf-8",
        )
        expect_raises(
            ValueError,
            lambda: LongV2VManifestDataset(str(missing_field_manifest)),
            "missing fields: reverse_prompt",
        )


def test_bidirectional_pair_dataset():
    def encode_npy(array):
        payload = io.BytesIO()
        np.save(payload, array, allow_pickle=False)
        return payload.getvalue()

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source = np.arange(3 * 4 * 4 * 3, dtype=np.uint8).reshape(3, 4, 4, 3)
        target = source + 32
        with ZipFile(root / "pairs.zip", "w") as archive:
            archive.writestr("source.npy", encode_npy(source))
            archive.writestr("target.npy", encode_npy(target))

        (root / "manifest.jsonl").write_text(
            json.dumps(
                {
                    "id": "sample-1",
                    "shard": "pairs.zip",
                    "source_member": "source.npy",
                    "target_member": "target.npy",
                    "prompt": "apply an edit",
                    "width": 4,
                    "height": 4,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        dataset_utils.SANA_STREAMING_TEST_RATIOS = {"1.0": [4, 4]}
        dataset = sana_v2v_pair_data.SanaV2VPairDataset(
            {"fixtures": str(root)},
            num_frames=3,
            aspect_ratio_type="SANA_STREAMING_TEST_RATIOS",
            max_retries=1,
        )
        sample = dataset[0]
        target_tensor, source_tensor = sample[0], sample[8]
        assert target_tensor.shape == source_tensor.shape == (3, 3, 4, 4)
        torch.testing.assert_close(
            target_tensor - source_tensor,
            torch.full_like(source_tensor, 64.0 / 255.0),
        )
        assert sample[1] == "apply an edit"
        assert sample[3]["cache_key"] == "fixtures/sample-1"
        sana_v2v_pair_data._open_zip.cache_clear()


def test_read_video_from_path():
    original_imiter = video_transforms.iio.imiter
    frames = [np.full((4, 6, 3), value, dtype=np.uint8) for value in (0, 64, 128, 255)]
    try:
        video_transforms.iio.imiter = lambda path, plugin: iter(frames)
        video = video_transforms.read_video_from_path("unused.mp4", (2, 3), num_frames=3)
        assert video.shape == (3, 3, 2, 3)
        assert video.dtype == torch.float32
        assert torch.isfinite(video).all()
        assert video.min() >= -1.0 and video.max() <= 1.0

        video_transforms.iio.imiter = lambda path, plugin: iter(frames[:2])
        expect_raises(
            RuntimeError,
            lambda: video_transforms.read_video_from_path("short.mp4", (2, 3), num_frames=3),
            "returned 2 frames, expected at least 3",
        )
    finally:
        video_transforms.iio.imiter = original_imiter


def test_v2v_batch_retry_counter_is_local():
    trainer = LongSANATrainer.__new__(LongSANATrainer)
    trainer.v2v = True
    trainer.dataloader = iter({"batch_id": index} for index in range(11))
    outcomes = iter([None] * 10 + [torch.tensor(1.0)])
    trainer._encode_v2v_source_batch = lambda batch: next(outcomes)

    batch, conditioning = trainer._next_batch_with_v2v_conditioning()
    assert batch["batch_id"] == 10
    assert torch.equal(conditioning, torch.tensor(1.0))
    assert not hasattr(trainer, "skipped_batches_count")

    trainer.dataloader = iter([{"batch_id": 11}, {"batch_id": 12}])
    outcomes = iter([None, torch.tensor(2.0)])
    trainer._encode_v2v_source_batch = lambda batch: next(outcomes)
    batch, conditioning = trainer._next_batch_with_v2v_conditioning()
    assert batch["batch_id"] == 12
    assert torch.equal(conditioning, torch.tensor(2.0))

    trainer.dataloader = iter({"batch_id": index} for index in range(11))
    trainer._encode_v2v_source_batch = lambda batch: None
    expect_raises(
        RuntimeError,
        trainer._next_batch_with_v2v_conditioning,
        "Too many consecutive V2V batches skipped",
    )


def make_cache_pipeline(pipeline_type=SanaTrainingPipeline, *, v2v, num_blocks, num_cached_blocks, sink_token):
    pipeline = pipeline_type.__new__(pipeline_type)
    pipeline.v2v = v2v
    pipeline.num_model_blocks = num_blocks
    pipeline.num_cached_blocks = num_cached_blocks
    pipeline.sink_token = sink_token
    pipeline._full_history_softmax_cache = v2v and num_cached_blocks < 0
    pipeline.block_is_state_cached = [False] * num_blocks
    pipeline._chunk_indices = [0, 2, 4, 6, 8, 10]
    pipeline.state = {"chunk_indices": pipeline._chunk_indices}
    pipeline.kv_cache = None
    pipeline._spatial_hw = 1
    pipeline.num_frame_per_block = 2
    pipeline.dtype = torch.float32
    return pipeline


def test_training_pipeline_cache_helpers():
    legacy = make_cache_pipeline(v2v=False, num_blocks=2, num_cached_blocks=1, sink_token=False)
    legacy_cache = legacy.initialize_kv_cache(num_chunks=2)
    assert len(legacy_cache) == 2
    assert all(len(block_cache) == 3 for chunk in legacy_cache for block_cache in chunk)
    for block_id in range(legacy.num_model_blocks):
        legacy_cache[0][block_id] = [
            torch.tensor(float(block_id + 1)),
            torch.tensor(float(block_id + 2)),
            torch.tensor(float(block_id + 3)),
        ]
    prepared_legacy, cached_chunks, sink_frames, cached_frames = legacy.prepare_kv_cache(1)
    assert (cached_chunks, sink_frames, cached_frames) == (0, 0, 0)
    for block_id in range(legacy.num_model_blocks):
        assert len(prepared_legacy[block_id]) == 3
        assert torch.equal(prepared_legacy[block_id][0], legacy_cache[0][block_id][0])
        assert torch.equal(prepared_legacy[block_id][1], legacy_cache[0][block_id][1])
        assert torch.equal(prepared_legacy[block_id][2], legacy_cache[0][block_id][2])

    v2v = make_cache_pipeline(v2v=True, num_blocks=2, num_cached_blocks=2, sink_token=True)
    v2v.block_is_state_cached = [False, True]
    v2v_cache = v2v.initialize_kv_cache(num_chunks=4)
    assert all(len(block_cache) == 6 for chunk in v2v_cache for block_cache in chunk)

    for chunk_idx in range(3):
        value = float(chunk_idx + 1)
        qkv = torch.full((1, 1, 1, 2), value)
        tconv = torch.full((1, 1, 2, 1), value)
        v2v_cache[chunk_idx][0] = [qkv.clone(), qkv.clone(), qkv.clone(), None, None, tconv.clone()]
        v2v_cache[chunk_idx][1] = [qkv.clone(), qkv.clone(), None, None, None, tconv.clone()]

    sampler = SANAStreamingSampler.__new__(SANAStreamingSampler)
    sampler.num_cached_blocks = v2v.num_cached_blocks
    sampler.sink_token = v2v.sink_token
    sampler._fixed_rope_full_history_softmax_cache = False
    sampler.block_is_state_cached = list(v2v.block_is_state_cached)
    sampler._chunk_indices = list(v2v._chunk_indices)
    sampler._spatial_hw = v2v._spatial_hw
    sampler_cache = copy.deepcopy(v2v_cache)

    prepared_v2v, cached_chunks, sink_frames, cached_frames = v2v.prepare_kv_cache(3)
    prepared_sampler, sampler_chunks, sampler_sink, sampler_frames = sampler.accumulate_kv_cache(sampler_cache, 3)
    assert cached_chunks == 2
    assert sink_frames == 2
    assert cached_frames == 4
    assert (sampler_chunks, sampler_sink, sampler_frames) == (cached_chunks, sink_frames, cached_frames)
    assert len(prepared_v2v[0]) == 6 and len(prepared_v2v[1]) == 6
    for training_block, inference_block in zip(prepared_v2v, prepared_sampler):
        for training_slot, inference_slot in zip(training_block, inference_block):
            if training_slot is None:
                assert inference_slot is None
            else:
                torch.testing.assert_close(training_slot, inference_slot)
    assert torch.equal(prepared_v2v[0][0].flatten(), torch.tensor([1.0, 1.0, 3.0, 3.0]))
    assert torch.isfinite(prepared_v2v[0][0]).all()
    assert torch.equal(prepared_v2v[1][0], v2v_cache[2][1][0])
    assert all(item is None for item in v2v_cache[1][0])

    history_cache = [
        [
            [
                torch.full((1, 1, 1, 2), float(chunk_idx + 1)),
                torch.full((1, 1, 1, 2), float(chunk_idx + 1)),
                torch.full((1, 1, 1, 2), float(chunk_idx + 1)),
                None,
                None,
                torch.full((1, 1, 2, 1), float(chunk_idx + 1)),
            ],
            [None] * 6,
        ]
        for chunk_idx in range(2)
    ]
    v2v.kv_cache = copy.deepcopy(history_cache)
    v2v._full_history_softmax_cache = True
    sampler._fixed_rope_full_history_softmax_cache = True
    sampler_history_cache = copy.deepcopy(history_cache)
    v2v.promote_kv_cache(1)
    sampler._promote_fixed_rope_full_history_cache(sampler_history_cache, 1)
    for training_slot, inference_slot in zip(v2v.kv_cache[1][0], sampler_history_cache[1][0]):
        if training_slot is None:
            assert inference_slot is None
        else:
            torch.testing.assert_close(training_slot, inference_slot)
    assert all(item is None for item in v2v.kv_cache[0][0])
    assert all(item is None for item in sampler_history_cache[0][0])

    rope_start, rope_end, frame_index = v2v.cache_positions(
        chunk_idx=3,
        start_f=6,
        end_f=8,
        sink_num=sink_frames,
        num_cached_frames=cached_frames,
        device=torch.device("cpu"),
    )
    assert (rope_start, rope_end) == (0, 8)
    assert torch.equal(frame_index, torch.tensor([0, 1, 4, 5, 6, 7]))
    assert torch.isfinite(frame_index).all()

    conditioning = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6, 1, 1)
    sliced = SanaTrainingPipeline._slice_chunk_data_info(conditioning, 2, 5)
    assert torch.equal(sliced["image_vae_embeds"], conditioning[:, :, 2:5])
    assert SanaTrainingPipeline._slice_chunk_data_info(None, 0, 2) == {}
    expect_raises(
        ValueError,
        lambda: SanaTrainingPipeline._slice_chunk_data_info(conditioning.squeeze(-1), 0, 2),
        "must be [B, C, T, H, W]",
    )
    expect_raises(
        ValueError,
        lambda: SanaTrainingPipeline._slice_chunk_data_info(conditioning, 4, 7),
        "does not cover the requested chunk",
    )


class DummyScheduler:
    def add_noise(self, clean, noise, timestep):
        del timestep
        return clean + 0.1 * noise


class DummyV2VGenerator(torch.nn.Module):
    def __init__(self, num_blocks):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        self.num_blocks = num_blocks
        self.calls = []

    def forward(
        self,
        noisy_image_or_video,
        condition,
        timestep,
        start_f,
        end_f,
        frame_index,
        save_kv_cache,
        mask,
        kv_cache,
        data_info,
    ):
        del condition, timestep, frame_index, mask, kv_cache
        conditioning = data_info["image_vae_embeds"]
        assert conditioning.shape == noisy_image_or_video.shape
        self.calls.append(
            {
                "conditioning": conditioning.detach().clone(),
                "save_kv_cache": save_kv_cache,
                "start_f": start_f,
                "end_f": end_f,
            }
        )

        pred_x0 = noisy_image_or_video * self.scale + 0.25 * conditioning
        flow_pred = pred_x0 - noisy_image_or_video
        updated_cache = None
        if save_kv_cache:
            batch_size = noisy_image_or_video.shape[0]
            qkv = noisy_image_or_video.mean(dim=1).reshape(batch_size, 1, 1, -1)
            tconv = noisy_image_or_video.mean(dim=(1, 3, 4)).unsqueeze(1).unsqueeze(-1)
            updated_cache = [
                [qkv.clone(), qkv.clone(), qkv.clone(), None, None, tconv.clone()]
                for _ in range(self.num_blocks)
            ]
        return flow_pred, pred_x0, updated_cache


def test_training_pipeline_multi_chunk_conditioning():
    torch.manual_seed(3407)
    generator = DummyV2VGenerator(num_blocks=1)
    pipeline = make_cache_pipeline(
        v2v=True,
        num_blocks=1,
        num_cached_blocks=2,
        sink_token=True,
    )
    pipeline.generator = generator
    pipeline.scheduler = DummyScheduler()
    pipeline.denoising_step_list = [5]
    pipeline.same_step_across_blocks = True
    pipeline.last_step_only = False
    pipeline.update_kv_cache_by_end = False

    noise = torch.randn(1, 2, 4, 1, 1)
    conditioning = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4, 1, 1)
    prompt_embeds = torch.ones(1, 2, 3)
    output, timestep_from, timestep_to = pipeline.generate_chunk_with_cache(
        noise,
        prompt_embeds,
        image_vae_embeds=conditioning,
    )

    assert output.shape == noise.shape
    assert output.requires_grad
    assert (timestep_from, timestep_to) == (5, 5)
    assert len(generator.calls) == 4
    expected_conditioning = (
        conditioning[:, :, :2],
        conditioning[:, :, :2],
        conditioning[:, :, 2:],
        conditioning[:, :, 2:],
    )
    for call, expected in zip(generator.calls, expected_conditioning):
        torch.testing.assert_close(call["conditioning"], expected)
    assert [call["save_kv_cache"] for call in generator.calls] == [False, True, False, True]
    assert pipeline.kv_cache[0][0][0] is not None
    assert pipeline.kv_cache[1][0][0] is not None

    output.sum().backward()
    assert generator.scale.grad is not None
    assert torch.isfinite(generator.scale.grad)


def test_reverse_reg_pipeline():
    torch.manual_seed(3407)
    generator = DummyV2VGenerator(num_blocks=1)
    pipeline = make_cache_pipeline(
        SanaReverseRegPipeline,
        v2v=True,
        num_blocks=1,
        num_cached_blocks=2,
        sink_token=True,
    )
    pipeline.block_is_state_cached = [False]
    pipeline.generator = generator
    pipeline.scheduler = DummyScheduler()
    pipeline.denoising_step_list = [5]
    pipeline.same_step_across_blocks = True
    pipeline.last_step_only = True

    source = torch.randn(1, 2, 4, 1, 1)
    edited = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4, 1, 1).expand_as(source).clone()
    prompt_embeds = torch.ones(1, 2, 3)

    expect_raises(
        ValueError,
        lambda: pipeline.denoise_chunk(source, edited[:, :, :3], prompt_embeds),
        "must have identical shapes",
    )

    result = pipeline.denoise_chunk(source, edited, prompt_embeds)
    assert result["flow_pred"].shape == source.shape
    assert result["pred_x0"].shape == source.shape
    assert result["noise"].shape == source.shape
    assert result["timestep"].shape == (1, 2)
    assert torch.equal(result["timestep"], torch.full((1, 2), 5, dtype=torch.int64))
    assert all(torch.isfinite(value).all() for value in result.values())

    assert len(generator.calls) == 4
    expected_chunks = (edited[:, :, :2], edited[:, :, :2], edited[:, :, 2:], edited[:, :, 2:])
    for call, expected_conditioning in zip(generator.calls, expected_chunks):
        assert torch.equal(call["conditioning"], expected_conditioning)
    assert [call["save_kv_cache"] for call in generator.calls] == [False, True, False, True]

    loss = result["flow_pred"].square().mean() + result["pred_x0"].square().mean()
    loss.backward()
    assert generator.scale.grad is not None
    assert torch.isfinite(generator.scale.grad)
    assert generator.scale.grad.abs() > 0


class DummyReversePipeline:
    def __init__(self):
        self.calls = []

    def denoise_chunk(self, **kwargs):
        self.calls.append(kwargs)
        source = kwargs["source_chunk_bcfhw"]
        return {
            "flow_pred": torch.zeros_like(source),
            "pred_x0": torch.zeros_like(source),
            "timestep": torch.ones(source.shape[0], dtype=torch.int64),
            "noise": torch.zeros_like(source),
        }


def test_reverse_overlap_uses_only_new_frames():
    model = StreamingSANATrainingModel.__new__(StreamingSANATrainingModel)
    model.reverse_pipeline = DummyReversePipeline()
    model.state = {
        "conditional_info": {
            "reverse_conditional_dict": {
                "prompt_embeds": torch.ones(1, 2, 3),
                "mask": torch.ones(1, 2),
            }
        }
    }

    edited = torch.arange(11, dtype=torch.float32).reshape(1, 11, 1, 1, 1)
    source = torch.arange(11, dtype=torch.float32).reshape(1, 1, 11, 1, 1) + 100
    gradient_mask = torch.zeros_like(edited, dtype=torch.bool)
    gradient_mask[:, 2:] = True
    result = model.run_reverse_denoise(
        edited,
        {
            "chunk_image_vae_embeds": source,
            "gradient_mask": gradient_mask,
            "new_frames_generated": 9,
            "chunk_start_frame": 11,
        },
    )

    call = model.reverse_pipeline.calls[0]
    assert call["current_start_frame"] == 11
    assert torch.equal(call["edited_chunk_bcfhw"], edited[:, -9:].permute(0, 2, 1, 3, 4))
    assert torch.equal(call["source_chunk_bcfhw"], source[:, :, -9:])
    assert result["gradient_mask"].shape[1] == 9
    assert result["gradient_mask"].all()


class DummyTextEncoder:
    def forward_chi(self, prompts, use_chi_prompt=True):
        del use_chi_prompt
        batch_size = len(prompts)
        return {
            "prompt_embeds": torch.ones(batch_size, 2, 3),
            "mask": torch.ones(batch_size, 2),
        }


def test_v2v_inference_pipeline_multi_chunk_conditioning():
    torch.manual_seed(3407)
    generator = DummyV2VGenerator(num_blocks=1)
    pipeline = SanaInferencePipeline.__new__(SanaInferencePipeline)
    pipeline.args = type("Args", (), {"motion_score": 0})()
    pipeline.device = torch.device("cpu")
    pipeline.generator = generator
    pipeline.text_encoder = DummyTextEncoder()
    pipeline.vae = None
    pipeline.scheduler = DummyScheduler()
    pipeline.num_frame_per_block = 2
    pipeline.denoising_step_list = [5]
    pipeline.model_device = torch.device("cpu")
    pipeline.model_dtype = torch.float32
    pipeline.v2v = True
    pipeline.num_model_blocks = 1
    pipeline.num_cached_blocks = 2
    pipeline.sink_token = True
    pipeline._full_history_softmax_cache = False
    pipeline.block_is_state_cached = [False]

    noise = torch.randn(1, 2, 4, 1, 1)
    conditioning = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4, 1, 1)
    output, info = pipeline.inference(
        noise,
        text_prompts=["apply an edit"],
        data_info={"image_vae_embeds": conditioning},
        generator=torch.Generator(device="cpu").manual_seed(3407),
    )

    assert output.shape == (1, 4, 2, 1, 1)
    assert info["chunk_indices"] == [0, 2, 4]
    assert len(generator.calls) == 4
    expected_conditioning = (
        conditioning[:, :, :2],
        conditioning[:, :, :2],
        conditioning[:, :, 2:],
        conditioning[:, :, 2:],
    )
    for call, expected in zip(generator.calls, expected_conditioning):
        torch.testing.assert_close(call["conditioning"], expected)
    assert [call["save_kv_cache"] for call in generator.calls] == [False, True, False, True]


test_release_configs()
test_long_v2v_manifest_dataset()
test_bidirectional_pair_dataset()
test_read_video_from_path()
test_v2v_batch_retry_counter_is_local()
test_training_pipeline_cache_helpers()
test_training_pipeline_multi_chunk_conditioning()
test_reverse_reg_pipeline()
test_reverse_overlap_uses_only_new_frames()
test_v2v_inference_pipeline_multi_chunk_conditioning()
print("SANA-Streaming CPU training helper tests passed")
PY

echo "Testing one SANA-Streaming bidirectional V2V training step"

python - <<'PY'
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import yaml

data_dir = Path("data/sana_streaming_training_ci/bidirectional")
output_dir = Path("output/test_sana_streaming_training_ci/bidirectional")
data_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

num_frames = 17
height, width = 192, 352
source = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
source[..., 0] = np.arange(num_frames, dtype=np.uint8)[:, None, None]
source[..., 1] = np.arange(height, dtype=np.uint8)[None, :, None]
source[..., 2] = np.arange(width, dtype=np.uint16)[None, None, :] % 256
target = np.roll(source, shift=1, axis=-1)


def encode_npy(array):
    payload = io.BytesIO()
    np.save(payload, array, allow_pickle=False)
    return payload.getvalue()


with ZipFile(data_dir / "pairs.zip", "w") as archive:
    archive.writestr("source.npy", encode_npy(source))
    archive.writestr("target.npy", encode_npy(target))

manifest_rows = [
    {
        "id": f"ci-pair-{index}",
        "shard": "pairs.zip",
        "source_member": "source.npy",
        "target_member": "target.npy",
        "prompt": "Transform the scene into a watercolor painting.",
        "width": width,
        "height": height,
    }
    for index in range(8)
]
(data_dir / "manifest.jsonl").write_text(
    "\n".join(json.dumps(row) for row in manifest_rows) + "\n",
    encoding="utf-8",
)

config = yaml.safe_load(
    Path("configs/sana_streaming/train/sana_streaming_bidirectional_2b_720p.yaml").read_text(encoding="utf-8")
)
config["data"]["data_dir"] = {"ci": str(data_dir)}
config["data"]["image_size"] = 256
config["data"]["aspect_ratio_type"] = "ASPECT_RATIO_VIDEO_256_TEST_DIV32"
config["data"]["num_frames"] = num_frames
config["model"]["image_size"] = 256
config["model"]["aspect_ratio_type"] = "ASPECT_RATIO_VIDEO_256_TEST_DIV32"
config["train"]["num_epochs"] = 1
config["train"]["num_workers"] = 0
config["train"]["log_interval"] = 1
config["train"]["save_model_epochs"] = 100
config["train"]["save_model_steps"] = 100
config["train"]["work_dir"] = str(output_dir / "run")
config["work_dir"] = str(output_dir / "run")
config["debug"] = True
config["report_to"] = "tensorboard"

config_path = output_dir / "sana_streaming_bidirectional_ci.yaml"
config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print(f"Wrote {config_path}")
PY

DISABLE_XFORMERS=1 torchrun --nproc_per_node=8 --master_port=$((RANDOM % 10000 + 20000)) \
    train_video_scripts/train_video_ivjoint_chunk.py \
    --config_path="$TEST_OUTPUT_DIR/bidirectional/sana_streaming_bidirectional_ci.yaml" \
    2>&1 | tee "$TEST_OUTPUT_DIR/bidirectional/train.log"

python - <<'PY'
import math
import re
from pathlib import Path

log = Path("output/test_sana_streaming_training_ci/bidirectional/train.log").read_text(encoding="utf-8")
assert "Global Step: 1" in log
match = re.search(r"loss:([-+0-9.eE]+)", log)
assert match is not None, "Missing loss in bidirectional training log"
assert math.isfinite(float(match.group(1))), f"Non-finite bidirectional loss: {match.group(1)}"
print("SANA-Streaming bidirectional V2V training step passed")
PY

echo "Testing one SANA-Streaming long V2V training step"

python - <<'PY'
import json
import shutil
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import yaml

data_dir = Path("data/sana_streaming_training_ci")
output_dir = Path("output/test_sana_streaming_training_ci")
shutil.rmtree(data_dir, ignore_errors=True)
shutil.rmtree(output_dir, ignore_errors=True)
data_dir.mkdir(parents=True)
output_dir.mkdir(parents=True)

num_frames = 81
height = width = 64
frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
frames[..., 0] = np.arange(num_frames, dtype=np.uint8)[:, None, None]
frames[..., 1] = np.arange(height, dtype=np.uint8)[None, :, None]
frames[..., 2] = np.arange(width, dtype=np.uint8)[None, None, :]
source_video = data_dir / "source.mp4"
iio.imwrite(source_video, frames, plugin="pyav", fps=16, codec="libx264")

manifest = data_dir / "manifest.jsonl"
manifest.write_text(
    json.dumps(
        {
            "prompt": "Transform the scene into a watercolor painting.",
            "reverse_prompt": "Restore the watercolor scene to the source video.",
            "source_video": source_video.name,
        }
    )
    + "\n",
    encoding="utf-8",
)

config_path = Path("configs/sana_streaming/train/sana_streaming_long_441_2b_720p.yaml")
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
config["generator_ckpt"] = "hf://Efficient-Large-Model/SANA-Streaming/dit/sana_streaming_ar.pth"
config["fake_ckpt"] = (
    "hf://Efficient-Large-Model/SANA-Streaming_bidirectional/dit/sana_bidirectional_short.pth"
)
config["real_ckpt"] = config["fake_ckpt"]
config["data_path"] = str(manifest)
config["data_root"] = str(data_dir)
config["batch_size"] = 1
config["log_iters"] = 1
config["image_or_video_shape"] = [1, 11, 128, 2, 2]
config["streaming_chunk_size"] = 11
config["streaming_max_length"] = 11
config["student_max_frame"] = 11
config["streaming_min_new_frame"] = 9
config["num_frame_per_block"] = 3
config["num_chunks_per_clip"] = 3

ci_config = output_dir / "sana_streaming_long_ci.yaml"
ci_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print(f"Wrote {ci_config}")
PY

DISABLE_XFORMERS=1 torchrun --nproc_per_node=8 --master_port=$((RANDOM % 10000 + 20000)) \
    train_video_scripts/train_longsana.py \
    --config_path "$TEST_OUTPUT_DIR/sana_streaming_long_ci.yaml" \
    --logdir "$TEST_OUTPUT_DIR/run" \
    --disable-wandb --no-auto-resume --no_save --no_visualize --max_iters=0 \
    2>&1 | tee "$TEST_OUTPUT_DIR/train.log"

python - <<'PY'
import math
import re
from pathlib import Path

log = Path("output/test_sana_streaming_training_ci/train.log").read_text(encoding="utf-8")
assert "'step': 1" in log
for key in ("generator_loss", "reverse_reg_loss", "reverse_reg_weighted_loss", "critic_loss"):
    match = re.search(rf"'{key}': '([^']+)'", log)
    assert match is not None, f"Missing {key} in training log"
    assert math.isfinite(float(match.group(1))), f"Non-finite {key}: {match.group(1)}"
print("SANA-Streaming long V2V training step passed")
PY
