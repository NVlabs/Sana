from __future__ import annotations

import io
import json
from zipfile import ZipFile

import numpy as np

from diffusion.data.datasets.video.sana_wm_zip_latent_data import SanaWMZipLatentDataset


def _npz_bytes(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def test_sana_wm_zip_latent_dataset_uses_external_caption(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    cache_dir = tmp_path / "cache"
    raw_dir.mkdir()
    cache_dir.mkdir()

    raw_zip = raw_dir / "sample.zip"
    cache_zip = cache_dir / "sample.zip"
    key = "clip_000"
    with ZipFile(raw_zip, "w") as z:
        z.writestr(
            f"{key}.json",
            json.dumps({"prompt": "zip prompt", "height": 64, "width": 64}),
        )
    with ZipFile(cache_zip, "w") as z:
        z.writestr(f"{key}.npz", _npz_bytes(z=np.zeros((1, 1, 2, 2), dtype=np.float32)))

    suffix = "_LongSceneStaticCaption-Qwen3-VL-30B-A3B-Instruct"
    (raw_dir / f"sample{suffix}.json").write_text(
        json.dumps({key: {"prompt": "external static caption"}}),
        encoding="utf-8",
    )

    dataset = SanaWMZipLatentDataset(
        data_dir={"sekai_game": str(raw_dir)},
        vae_cache_dir=str(cache_dir),
        min_latent_file_size=0,
        external_caption_suffixes=[suffix],
        caption_proportion={"sekai_game": {suffix: 100}},
    )

    sample = dataset[0]
    assert sample[1] == "external static caption"
    assert sample[3]["caption_type"] == suffix


def test_sana_wm_zip_latent_dataset_filters_and_repeats(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    cache_dir = tmp_path / "cache"
    raw_dir.mkdir()
    cache_dir.mkdir()

    raw_zip = raw_dir / "sample.zip"
    cache_zip = cache_dir / "sample.zip"
    keys = ("keep", "drop")
    with ZipFile(raw_zip, "w") as z:
        for key in keys:
            z.writestr(f"{key}.json", json.dumps({"prompt": key, "height": 64, "width": 64}))
    with ZipFile(cache_zip, "w") as z:
        for key in keys:
            z.writestr(f"{key}.npz", _npz_bytes(z=np.zeros((1, 1, 2, 2), dtype=np.float32)))

    (raw_dir / "sample_vmafmotion.json").write_text(
        json.dumps({"keep": {"vmafmotion_score": 1.0}, "drop": {"vmafmotion_score": 0.1}}),
        encoding="utf-8",
    )

    dataset = SanaWMZipLatentDataset(
        data_dir={"sekai_game": str(raw_dir)},
        vae_cache_dir=str(cache_dir),
        min_latent_file_size=0,
        external_data_filter={"sekai_game": {"_vmafmotion": {"min": 0.5, "max": 50}}},
        data_repeat={"sekai_game": 3},
    )

    assert len(dataset) == 3
    assert {dataset[idx][3]["key"] for idx in range(len(dataset))} == {"keep"}


def test_sana_wm_zip_latent_dataset_applies_metric_scale_before_relative_pose(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    cache_dir = tmp_path / "cache"
    raw_dir.mkdir()
    cache_dir.mkdir()

    raw_zip = raw_dir / "sample.zip"
    cache_zip = cache_dir / "sample.zip"
    key = "clip_000"
    with ZipFile(raw_zip, "w") as z:
        z.writestr(f"{key}.json", json.dumps({"prompt": key, "height": 64, "width": 64}))
    with ZipFile(cache_zip, "w") as z:
        z.writestr(f"{key}.npz", _npz_bytes(z=np.zeros((1, 2, 2, 2), dtype=np.float32)))

    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    poses[1, 0, 3] = 1.0
    intrinsics = np.array([[64, 64, 32, 32], [64, 64, 32, 32]], dtype=np.float32)
    np.savez(
        raw_dir / "sample_camera.npz",
        ids=np.array([key]),
        ranges=np.array([[0, 2]], dtype=np.int64),
        pose=poses,
        intrinsics=intrinsics,
    )
    (raw_dir / "sample_metric_scale_stats.json").write_text(
        json.dumps({key: {"median": 2.0}}),
        encoding="utf-8",
    )

    dataset = SanaWMZipLatentDataset(
        data_dir={"sekai_game": str(raw_dir)},
        vae_cache_dir=str(cache_dir),
        min_latent_file_size=0,
        vae_ratio=(1, 32),
    )

    camera_conditions = dataset[0][6]
    assert camera_conditions.shape == (2, 20)
    assert camera_conditions[1, 3].item() == 2.0
