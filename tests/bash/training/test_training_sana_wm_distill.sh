#!/bin/bash
set -e

echo "Testing the SANA-WM ODE and self-forcing distillation chain"

cleanup() {
    rm -rf data/sana_wm_distill_ci output/test_sana_wm_distill_ci
}
trap cleanup EXIT

python - <<'PY'
import io
import json
import shutil
import zipfile
from pathlib import Path

import lmdb
import numpy as np
import yaml

from diffusion.longsana.utils.lmdb import store_arrays_to_lmdb


data_root = Path("data/sana_wm_distill_ci")
ode_data_root = data_root / "ode_trajectories"
raw_root = data_root / "sekai_game"
cache_root = data_root / "vae_cache"
out_root = Path("output/test_sana_wm_distill_ci")

for path in (data_root, out_root):
    shutil.rmtree(path, ignore_errors=True)
ode_data_root.mkdir(parents=True)
raw_root.mkdir(parents=True)
cache_root.mkdir(parents=True)
out_root.mkdir(parents=True)

# ODE uses DP8 in CI, so provide one deterministic trajectory per rank. Seven
# latent frames keep this smoke test short while still crossing chunk boundaries.
rng = np.random.default_rng(3407)
num_samples = 8
num_snapshots = 5
num_latent_frames = 7
latent_shape = (num_samples, num_latent_frames, 128, 22, 40)
clean = rng.standard_normal(latent_shape, dtype=np.float32)
noise = rng.standard_normal(latent_shape, dtype=np.float32)
sigmas = np.array([1.0, 0.967, 0.908, 0.764, 0.0], dtype=np.float32)
assert len(sigmas) == num_snapshots
latents = np.stack([(1.0 - sigma) * clean + sigma * noise for sigma in sigmas], axis=1).astype(np.float16)

camera_conditions = np.zeros((num_samples, num_latent_frames, 20), dtype=np.float16)
camera_conditions[..., :16] = np.eye(4, dtype=np.float16).reshape(1, 1, 16)
camera_conditions[..., 16:] = np.array([40.0, 22.0, 20.0, 11.0], dtype=np.float16)
chunk_plucker = np.zeros((num_samples, 48, num_latent_frames, 22, 40), dtype=np.float16)
prompts = np.array(["A car drives forward across a dry lake bed under a blue sky."] * num_samples)

ode_arrays = {
    "latents": latents,
    "prompts": prompts,
    "camera_conditions": camera_conditions,
    "chunk_plucker": chunk_plucker,
}
env = lmdb.open(str(ode_data_root), map_size=1 << 30)
with env.begin(write=True) as txn:
    for name, array in ode_arrays.items():
        txn.put(f"{name}_shape".encode(), " ".join(map(str, array.shape)).encode())
store_arrays_to_lmdb(env, ode_arrays)
env.sync()
env.close()

# Reuse the same zip-latent fixture layout as the existing Stage-1 CI test.
# The two self-forcing configs slice this 25-frame latent to 10 and 13 frames.
key = "sample_000000"
raw_zip = raw_root / "sekai_game_train_00000000.zip"
cache_zip = cache_root / raw_zip.name
prompt = "A car drives forward across a dry lake bed under a blue sky."

with zipfile.ZipFile(raw_zip, "w", compression=zipfile.ZIP_STORED) as zf:
    zf.writestr(f"{key}.json", json.dumps({"prompt": prompt, "width": 1280, "height": 704}))

latent = rng.standard_normal((128, 25, 22, 40), dtype=np.float32)
buf = io.BytesIO()
np.savez(buf, z=latent)
with zipfile.ZipFile(cache_zip, "w", compression=zipfile.ZIP_STORED) as zf:
    zf.writestr(f"{key}.npz", buf.getvalue())

num_pixel_frames = 193
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
raw_zip.with_name(raw_zip.stem + f"{caption_suffix}.json").write_text(
    json.dumps({key: {"prompt": prompt}}), encoding="utf-8"
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
        json.dumps({key: {"score": score}}), encoding="utf-8"
    )


def configure_train(cfg, work_dir, save_model_steps):
    cfg["work_dir"] = str(work_dir)
    cfg["train"]["batch_size"] = 1
    cfg["train"]["num_workers"] = 0
    cfg["train"]["max_steps"] = 1
    cfg["train"]["log_interval"] = 1
    cfg["train"]["save_model_steps"] = save_model_steps
    cfg["train"]["early_stop_hours"] = 0
    cfg["report_to"] = "none"
    cfg["resume_from"] = None


def configure_self_forcing_data(cfg):
    cfg["data"]["hf_dataset_repo"] = None
    cfg["data"]["hf_dataset_revision"] = None
    cfg["data"]["hf_dataset_local_dir"] = "."
    cfg["data"]["hf_dataset_allow_patterns"] = None
    cfg["data"]["data_dir"] = {"sekai_game": str(raw_root)}
    cfg["data"]["vae_cache_dir"] = str(cache_root)
    # CP4 on eight GPUs leaves DP2, so repeat the fixture once per DP rank.
    cfg["data"]["data_repeat"] = {"sekai_game": 2}
    cfg["data"]["num_frames"] = num_pixel_frames
    cfg["data"]["sort_dataset"] = True
    cfg["data"]["shuffle_dataset"] = False


ode_work_dir = out_root / "ode_t7"
ode_cfg = yaml.safe_load(Path("configs/sana_wm/distill/ode_t43.yaml").read_text(encoding="utf-8"))
ode_cfg["data_path"] = str(ode_data_root)
ode_cfg["max_samples"] = num_samples
ode_cfg["num_latent_frames"] = num_latent_frames
configure_train(ode_cfg, ode_work_dir, save_model_steps=1)
ode_ci_cfg = out_root / "ode_t7_ci.yaml"
ode_ci_cfg.write_text(yaml.safe_dump(ode_cfg, sort_keys=False), encoding="utf-8")

t43_work_dir = out_root / "self_forcing_t10"
t43_cfg = yaml.safe_load(Path("configs/sana_wm/distill/self_forcing_t43.yaml").read_text(encoding="utf-8"))
t43_cfg["model_path"] = str(ode_work_dir / "checkpoints/step_000001/model.pt")
# CP4 pads this to 12 frames, leaving three frames per rank. That is the
# minimum local length required by the temporal-convolution halo exchange.
t43_cfg["num_latent_frames"] = 10
configure_self_forcing_data(t43_cfg)
configure_train(t43_cfg, t43_work_dir, save_model_steps=1)
t43_ci_cfg = out_root / "self_forcing_t10_ci.yaml"
t43_ci_cfg.write_text(yaml.safe_dump(t43_cfg, sort_keys=False), encoding="utf-8")

t121_work_dir = out_root / "self_forcing_t13"
t121_cfg = yaml.safe_load(Path("configs/sana_wm/distill/self_forcing_t121.yaml").read_text(encoding="utf-8"))
t43_checkpoint = t43_work_dir / "checkpoints/step_000001/model.pt"
t121_cfg["model_path"] = str(t43_checkpoint)
t121_cfg["fake_model_path"] = str(t43_checkpoint)
# Four streaming chunks are enough to exercise the T121 sink + sliding-window
# cache policy without making CI roll out all 121 production frames.
t121_cfg["num_latent_frames"] = 13
configure_self_forcing_data(t121_cfg)
configure_train(t121_cfg, t121_work_dir, save_model_steps=0)
t121_ci_cfg = out_root / "self_forcing_t13_ci.yaml"
t121_ci_cfg.write_text(yaml.safe_dump(t121_cfg, sort_keys=False), encoding="utf-8")

print(f"Wrote {ode_ci_cfg}, {t43_ci_cfg}, and {t121_ci_cfg}")
PY

torchrun --nproc_per_node=8 --master_port=$((RANDOM % 10000 + 20000)) \
    train_video_scripts/train_longsana.py \
    --config_path output/test_sana_wm_distill_ci/ode_t7_ci.yaml \
    --disable-wandb --no-auto-resume --max_iters=1

ODE_CHECKPOINT=output/test_sana_wm_distill_ci/ode_t7/checkpoints/step_000001/model.pt
test -s "$ODE_CHECKPOINT"

torchrun --nproc_per_node=8 --master_port=$((RANDOM % 10000 + 20000)) \
    train_video_scripts/train_longsana.py \
    --config_path output/test_sana_wm_distill_ci/self_forcing_t10_ci.yaml \
    --disable-wandb --no-auto-resume --max_iters=1

T43_CHECKPOINT=output/test_sana_wm_distill_ci/self_forcing_t10/checkpoints/step_000001/model.pt
test -s "$T43_CHECKPOINT"

torchrun --nproc_per_node=8 --master_port=$((RANDOM % 10000 + 20000)) \
    train_video_scripts/train_longsana.py \
    --config_path output/test_sana_wm_distill_ci/self_forcing_t13_ci.yaml \
    --disable-wandb --no-auto-resume --no_save --max_iters=1

grep -q "mode=self_forcing step=1" output/test_sana_wm_distill_ci/self_forcing_t13/train_log.log
