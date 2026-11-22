# LIBERO evaluation

Closed-loop evaluation of SANA-Robot-AR LIBERO on [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus),
[LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO) and [LIBERO-MAX](https://github.com/liberomax/LIBERO-MAX).
The policy sees the two 256×256 cameras and the joint state, generates a chunk of 32 actions every 16 control steps
and keeps the frames of the whole episode in its memory.

## 1. Setup

Install the simulator packages into the SANA-Robot environment, and ImageMagick for the sensor noise of
LIBERO-Plus:

```bash
pip install -r eval/libero/requirements.txt
sudo apt-get install libmagickwand-dev
```

Get the benchmarks at the commits the numbers below were measured on:

```bash
git clone https://github.com/sylvestf/LIBERO-plus && git -C LIBERO-plus checkout 4976dc3
git clone https://github.com/Zxy-MLlab/LIBERO-PRO && git -C LIBERO-PRO checkout 2b910b5
git clone https://github.com/liberomax/LIBERO-MAX && git -C LIBERO-MAX checkout a1e3cef
```

- LIBERO-Plus: unpack `assets.zip` of [Sylvest/LIBERO-plus](https://huggingface.co/datasets/Sylvest/LIBERO-plus)
  into `LIBERO-plus/libero/libero/assets`.
- LIBERO-PRO: download [zhouxueyang/LIBERO-Pro](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro) to
  `LIBERO-Pro-data` (the folders `bddl_files` and `init_files`).

The checkpoint comes with the description of its training data and the normalization of its robot rows:

```
checkpoints/sana_robot_ar_libero/
├── sana_robot_ar_libero.pth
├── meta.json
└── normalization.json
```

## 2. LIBERO-Plus

10,030 perturbed tasks of the four suites, one episode each.

```bash
python -m eval.libero.evaluate --config_path eval/libero/deploy.yaml \
    --benchmark plus --libero_root LIBERO-plus --output output/libero_plus
python -m eval.libero.evaluate --config_path eval/libero/deploy.yaml \
    --benchmark plus --libero_root LIBERO-plus --output output/libero_plus --report true
```

An episode takes about 11 seconds on one H100, the benchmark about 30 GPU hours. To use several GPUs, start one
process per GPU with `--num_shards N --shard_index i`; a process skips the episodes that already have a result, so
a stopped run continues with the same command. The report prints the success rate over all episodes, per suite and
per perturbation category.

## 3. LIBERO-PRO

8,000 episodes: the object, position, semantic and task perturbations of the four suites, 10 tasks with 50 initial
states each.

```bash
python -m eval.libero.evaluate --config_path eval/libero/deploy.yaml \
    --benchmark pro --libero_root LIBERO-PRO --pro_data LIBERO-Pro-data --output output/libero_pro \
    --steps 10 --video_cfg_scale 3 --n_action_steps 8
```

With these settings (10 denoising steps, video guidance 3, a chunk every 8 control steps) an episode takes about
55 seconds; with the settings of LIBERO-Plus about 15 seconds.

## 4. LIBERO-MAX

8,000 pairs of episodes from one initial state: one without and one with the benchmark's change in the middle of
the episode. Run the cases that come from LIBERO-Plus and the ones that come from LIBERO-PRO, then score them with
the benchmark's aggregator:

```bash
python -m eval.libero.evaluate --config_path eval/libero/deploy.yaml \
    --benchmark max --max_root LIBERO-MAX --max_source plus --libero_root LIBERO-plus \
    --output output/libero_max_plus
python -m eval.libero.evaluate --config_path eval/libero/deploy.yaml \
    --benchmark max --max_root LIBERO-MAX --max_source pro --libero_root LIBERO-PRO --pro_data LIBERO-Pro-data \
    --output output/libero_max_pro
python LIBERO-MAX/scripts/aggregate_cosmos_benchmark.py output/libero_max_plus \
    --case-root output/libero_max_pro --manifest LIBERO-MAX/benchmark/max8000/libero_max_8000.json \
    --output-dir output/libero_max_report --query-interval 16 --require-render-qa
```

A pair takes about 30 seconds. The success rates are `end_to_end_metrics` of
`output/libero_max_report/benchmark_summary.json`: `control` without the change (Base) and `intervention` with it
(Dynamic).

## 5. Results

| Benchmark   | Success rate (%)              |
|-------------|-------------------------------|
| LIBERO-Plus | 93.98                         |
| LIBERO-PRO  | 68.09                         |
| LIBERO-MAX  | 88.30 (Base), 76.06 (Dynamic) |

LIBERO-Plus and LIBERO-PRO episodes may run for 400 control steps (700 on LIBERO-10). Within the 220 / 280 / 300 /
520 steps of the LIBERO suites the LIBERO-Plus success rate is 93.35 and the LIBERO-PRO success rate is 65.36; the
report prints both, and `--relaxed_step_limits false` stops the episodes at those steps. LIBERO-MAX always uses
them.
