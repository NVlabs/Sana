# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SANA-Robot is a 4.5B world action model: one diffusion transformer denoises the future video latents of every
camera and the robot's action chunk together, each at its own noise level. The package `sana_robot` holds two
policies:

- **SANA-Robot-AR** (`policy/sana_robot_ar.py`): autoregressive over action chunks with a memory of the episode.
  It is the released policy: every recipe, checkpoint, evaluation and README section is about it.
- **SANA-Robot-base** (`policy/sana_robot_base.py`): one chunk from the current observation, no memory. Its code
  (dataset, trainer, session) is in the tree; no recipe or checkpoint ships.

Both share two frozen models, expected under `checkpoints/`: the LTX-2.3 video VAE (`LTX-2.3-Diffusers/vae`) and
the Gemma 2 2B text encoder (`gemma-2-2b-it`). README.md is the user guide; `eval/*/README.md` and
`sana_robot/deploy/README.md` hold the details of evaluation and deployment.

## Commands

All commands run from the repository root. Entries are modules (`python -m ...`); `scripts/` holds thin shell
wrappers around them. The training, deployment and evaluation entries read their config from
`--config_path file.yaml`, and any field can be set on the command line (`--data.chunk_budget 16`,
`--instructions "['task a', 'task b']"`); the data tools and the robot client take plain flags. pyrallis cannot
read the config of an entry whose module has `from __future__ import annotations`.

```bash
# environment: Python 3.11, PyTorch 2.9.1 + CUDA 12.8, flash-attn compiled from source (about 40 minutes)
bash environment_setup.sh conda sana-robot        # or: bash environment_setup.sh venv .venv

# data: a benchmark's public release -> zip archive dataset + normalization file
bash scripts/data/convert_robodojo.sh /path/to/RoboDojo_lerobot_v30_video data/RoboDojo
bash scripts/data/convert_robotwin.sh /path/to/robotwin2.0 data/RoboTwin
bash scripts/data/convert_libero.sh /path/to/modified_libero_rlds data/LIBERO
python -m sana_robot.data.utils.compute_normalization --dataset <dataset> --output <normalization.json> \
    [--lerobot-mapping <mapping.json>]

# finetune SANA-Robot-AR. The recipes are written for nodes of 8 H100: 8 nodes for RoboDojo, 4 nodes for
# RoboTwin 2.0, 2 nodes for LIBERO (README section 4.1). On every node:
torchrun --nnodes 8 --nproc_per_node 8 --rdzv_id 1 --rdzv_backend c10d --rdzv_endpoint <first node>:29500 \
    -m sana_robot.train.train_ar --config_path configs/robodojo_ar.yaml
# one node (GPUS, 8 by default); scale data.chunk_budget so that a step trains the same number of chunks
bash scripts/train/train_ar.sh configs/libero_ar.yaml --data.chunk_budget 22

# optional: encode the video latents once and train from them (no image augmentation then)
bash scripts/data/build_robodojo_latents.sh data/RoboDojo checkpoints/LTX-2.3-Diffusers
bash scripts/train/train_ar.sh configs/robodojo_ar.yaml --data.latent_store data/RoboDojo/latents

# closed-loop evaluation (each benchmark needs its own simulator setup, see its README)
python -m eval.libero.evaluate --config_path eval/libero/deploy.yaml --benchmark plus --libero_root LIBERO-plus \
    --output output/libero_plus                    # add --report true to print the results
python -m eval.robotwin.server --config_path eval/robotwin/deploy.yaml --port 8000

# deployment: policy server on the GPU machine, control loop on the robot's machine
bash scripts/deploy/server.sh my_deploy.yaml --host 0.0.0.0 --port 8000
python sana_robot/deploy/client.py --host <GPU machine> --port 8000 --instruction "..."
bash scripts/deploy/benchmark.sh my_deploy.yaml [--optimized false] [--precision fp8]

# formatting of the files you changed (pyproject.toml: line length 120, isort profile black)
black <files> && isort <files>
```

There is no test suite in the repository. To exercise a change end to end, run the entry it touches on a small
setting, for example a short training run or the latency benchmark on the plain PyTorch path:

```bash
WANDB_MODE=offline GPUS=2 bash scripts/train/train_ar.sh configs/robodojo_ar.yaml \
    --work_dir output/check --max_steps 12 --save_steps 6 --log_steps 1 --data.chunk_budget 40
bash scripts/deploy/benchmark.sh my_deploy.yaml --optimized false
```

A training run resumes from the newest checkpoint in its `work_dir` (`checkpoints/step_<n>/`, with
`checkpoints/latest` pointing at the last complete one). `model_ema.pth` in a checkpoint holds the averaged
weights, which is the file a deploy config names as `checkpoint`.

## Architecture

### Vocabulary shared by every part

- **Robot row**: every state and every action is a row of 80 slots with fixed named fields (joints, end-effector
  position, 6D rotation, gripper, hand; left and right arm), defined in `data/format/robot80.py`. A dataset fills
  only some slots and says which with its `state_mask` and `action_mask`. Grippers are stored as closedness in
  [0, 1]. The policy sees rows normalized with the dataset's 1% and 99% quantiles (`normalization.json`; the
  exact rule is in `data/format/normalization.py`).
- **Control step, chunk, video frame**: an action chunk is `chunk_size` control steps (32); the video has a frame
  every `video_frame_stride` control steps (4); the VAE turns 8 frames into one latent frame.
- **Views**: every camera has a slot in the spatial layout (`policy.spatial_layout`: `t_shape` or `2x2`) and its
  own pixel size (`view_sizes`); the latents of all views of one latent frame are concatenated into one token
  strip. `view_sizes` must be the sizes the checkpoint was trained with; the default (three views of 256x320) is
  not what the recipes use.
- **Prompts**: one text per view and one for the robot rows, built from the dataset's embodiment, action mode
  and camera descriptions plus the instruction (`data/format/prompt.py`).
- **DatasetInfo** (`data/format/episode.py`): control frequency, embodiment, action mode, views and the two
  masks. Training and deployment both read it from the dataset, so a deploy config names, besides the weights,
  the training dataset (its `meta.json` is enough for a zip archive dataset; a LeRobot dataset needs its
  directory and mapping file) and its normalization file.

### Data

`data/storage/open_storage` returns a reader with the same interface (`info`, `episodes`, `read_rows`,
`read_frames`) for the two formats: zip archives written by the converters in `data/utils/`, and LeRobot v2.x /
v3.0 datasets read in place through a mapping file from columns to robot fields (`data/storage/lerobot.py`
documents the mapping). `data/ar/` builds whole-episode windows for SANA-Robot-AR and `data/base/` one-chunk
windows for SANA-Robot-base. A `LatentStore` replaces frame decoding and VAE encoding by precomputed latents.
To add a robot, either write a converter like `data/utils/convert_robodojo.py` (it shows forward kinematics from
joints into the end-effector fields) or use a LeRobot dataset with a mapping.

### Training SANA-Robot-AR

`train/train_ar.py` wires the dataset, the encoders and the step function into the loop of
`train/runtime/loop.py` (Accelerate, bf16 autocast, FSDP that shards the parameters inside a node and replicates
them across nodes, averaged weights, gradient clipping per parameter group).

- A training sample is a **window**: an episode from its first frame to its end, cut into chunks; every episode
  gives one window per phase of its first row (`data/ar/windows.py`).
- `data/ar/packing/` orders the windows so that every stretch keeps the task mix, then packs them into one row
  of windows per GPU with at most `data.chunk_budget` chunks. The chunk budget is the GPU memory knob. The plan
  depends only on the windows, the seed, the epoch and the number of GPUs, which is what lets a run resume inside
  an epoch.
- The policy runs a window as the **teacher-forcing sequence** [sink | clean chunks | noisy chunks]
  (`data/ar/teacher_forcing.py`, `policy/layout/teacher_forcing.py`): the sink is the first frame, a noisy chunk
  reads the clean chunks before it, and every chunk has its own timestep. The loss is the flow-matching loss of
  the video latents and the actions, averaged over the trained chunks of all GPUs (`train/step/ar.py`,
  `train/flow_matching/`).
- The collate pads the windows of a row to its longest one only to stack tensors; with `chunk_counts` in the
  batch the policy (`SanaRobotAR._forward_packed`) lays the real chunks of all windows end to end in one token
  sequence, in which attention, the recurrent state and the positions restart at every window.

### Policy

`policy/backbone/dit.py` is the shared transformer: 32 layers over video, state and action tokens, a quarter of
them softmax attention and the rest Gated DeltaNet layers (linear attention with a recurrent state), with text
cross-attention and rotary positions over time and the spatial layout. `SanaRobotAR` gives it memory: the softmax
layers read the sink and the clean latents of the last `sliding_window_chunks` chunks, the Gated DeltaNet layers
carry one state over all past chunks. It has two faces that must stay equivalent: the training forward on the
teacher-forcing sequence, and the streaming methods (`forward_sink`, `forward_observation`, `forward_chunk`)
that inference drives through per-layer caches.

`kernels/` holds every hand-written kernel: `kernels/training/` is what the training forward imports,
`kernels/inference/` is installed on a policy by `install_kernels` when a deployment is `optimized`
(per-GPU kernels for H100, RTX 5090, RTX PRO 6000 and Jetson Thor; `bf16`, `fp8` and `nvfp4` precision).
Other GPUs run the plain PyTorch path.

### Deployment

`deploy/runtime/deployment.py` (`DeployConfig`, `Deployment`) loads the policy and the two encoders once. On
top of it:

- `deploy/session/`: streaming inference over one episode (`ARSession`: open the memory with the sink, commit
  the chunk the robot executed, generate the next one; optional classifier-free guidance of the video). With
  `optimized: true` commits and generations replay as CUDA graphs (`deploy/runtime/cuda_graph.py`), which are
  captured in the first requests of a task.
- `deploy/memory/`: the softmax memory, and `ViewStream`, which encodes the camera frames block by block and
  gives the same latents as the training encoding of the whole episode.
- `deploy/async_exec/`: `ChunkRunner` is the interface a control loop drives (`start`, `push_frame`,
  `next_plan`); the policy's work runs on one worker thread. `async_delay_steps` makes execution asynchronous;
  SANA-Robot-AR then keeps one memory per execution phase (`memory_rtc.py`) so that every request reads a memory
  laid out as in training.
- `deploy/server.py` and `deploy/client.py`: the policy server and the robot-side template. The client needs
  only NumPy and OpenCV, resizes the frames to the policy's resolution and speaks a small TCP protocol (JSON
  header plus raw arrays). A user fills in `Robot.read` and `Robot.send`; the server's `RobotAdapter` converts
  between the robot's vectors and robot rows. `deploy/yam/` is a complete example for two YAM arms.

### Evaluation

`eval/policy.py` (`EpisodePolicy`) is the closed-loop wrapper LIBERO and RoboTwin share. `eval/robodojo/` is an
XPolicyLab adapter that is copied into a RoboDojo checkout; its `__init__.py`, `deploy.py`, `eval.sh` and the two
`setup_eval_*.sh` files are XPolicyLab's own files and must stay byte for byte as they are (do not reformat
them). The released evaluation settings live in each directory's `deploy.yaml` / `deploy.yml`.
