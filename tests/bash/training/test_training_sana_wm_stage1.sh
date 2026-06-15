#!/bin/bash
set -e

echo "Testing SANA-WM stage-1 chunk-causal training"

python - <<'PY'
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import yaml

root = Path("data/sana_wm_stage1_ci")
cache_root = Path("data/sana_wm_stage1_ci_vae_cache")
out_dir = Path("output/test_sana_wm_stage1_ci")
root.mkdir(parents=True, exist_ok=True)
cache_root.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)

key = "sample_000000"
raw_zip = root / "sekai_game_train_00000000.zip"
cache_zip = cache_root / raw_zip.name

with zipfile.ZipFile(raw_zip, "w", compression=zipfile.ZIP_STORED) as zf:
    zf.writestr(
        f"{key}.json",
        json.dumps(
            {
                "prompt": "A car drives forward across a dry lake bed under a blue sky.",
                "width": 1280,
                "height": 704,
            }
        ),
    )

rng = np.random.default_rng(3407)
latent = rng.standard_normal((128, 24, 22, 40), dtype=np.float32)
buf = io.BytesIO()
np.savez(buf, z=latent)
with zipfile.ZipFile(cache_zip, "w", compression=zipfile.ZIP_STORED) as zf:
    zf.writestr(f"{key}.npz", buf.getvalue())

num_pixel_frames = 192
poses = np.repeat(np.eye(4, dtype=np.float32)[None], num_pixel_frames, axis=0)
poses[:, 2, 3] = np.linspace(0.0, 1.0, num_pixel_frames, dtype=np.float32)
intrinsics = np.tile(np.array([760.0, 760.0, 640.0, 352.0], dtype=np.float32), (num_pixel_frames, 1))
np.savez(
    raw_zip.with_name(raw_zip.stem + "_camera.npz"),
    ids=np.array([key]),
    ranges=np.array([[0, num_pixel_frames]], dtype=np.int64),
    pose=poses,
    intrinsics=intrinsics,
)

caption_suffix = "_LongSceneStaticCaption-Qwen3-VL-30B-A3B-Instruct"
(raw_zip.with_name(raw_zip.stem + f"{caption_suffix}.json")).write_text(
    json.dumps({key: {"prompt": "A car drives forward across a dry lake bed under a blue sky."}}),
    encoding="utf-8",
)

filter_scores = {
    "_vmafmotion": 1.0,
    "_unimatch": 10.0,
    "_dover": 0.5,
    "_vlm_entity_filter": 5.0,
    "_vlm_quality_filter": 1.0,
}
for suffix, score in filter_scores.items():
    raw_zip.with_name(raw_zip.stem + f"{suffix}.json").write_text(
        json.dumps({key: {"score": score}}),
        encoding="utf-8",
    )

cfg_path = Path("configs/sana_wm/stage1/sana_wm_stage1_sekai_chunk_causal_cp2_fsdp2.yaml")
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
cfg["data"]["hf_dataset_repo"] = None
cfg["data"]["hf_dataset_revision"] = None
cfg["data"]["hf_dataset_local_dir"] = "."
cfg["data"]["hf_dataset_allow_patterns"] = None
cfg["data"]["data_dir"] = {"sekai_game": str(root)}
cfg["data"]["vae_cache_dir"] = str(cache_root)
cfg["data"]["data_repeat"] = {"sekai_game": 1}
cfg["data"]["num_frames"] = num_pixel_frames
cfg["train"]["num_workers"] = 0
cfg["train"]["max_steps"] = 1
cfg["train"]["num_epochs"] = 1
cfg["train"]["log_interval"] = 1
cfg["train"]["save_model_steps"] = 0
cfg["train"]["work_dir"] = str(out_dir)
cfg["work_dir"] = str(out_dir)
cfg["report_to"] = "none"
cfg["name"] = "ci_sana_wm_stage1"

ci_cfg = out_dir / "sana_wm_stage1_ci.yaml"
ci_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(f"Wrote {ci_cfg}")
PY

torchrun --nproc_per_node=2 --master_port=$((RANDOM % 10000 + 20000)) \
    train_video_scripts/train_sana_wm_stage1.py \
    --config_path=output/test_sana_wm_stage1_ci/sana_wm_stage1_ci.yaml
