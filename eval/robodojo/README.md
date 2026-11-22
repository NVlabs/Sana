# RoboDojo evaluation

[RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) evaluates a policy through
[XPolicyLab](https://github.com/XPolicyLab/XPolicyLab), where every policy is an adapter under `policy/<NAME>/`.
This directory is the adapter of SANA-Robot, `SANA_Robot`: a policy server on the `sana_robot` package that takes
the observations of the simulator's ARX-X5 robot (`env_cfg_type=arx_x5`) and answers with joint targets
(`action_type=joint`) or end-effector poses (`action_type=ee`). One checkpoint serves both.

## 1. Setup

Install RoboDojo and its simulator environment as its documentation describes. Then copy this directory into its
XPolicyLab checkout under the name `SANA_Robot`, and install XPolicyLab into the SANA-Robot environment:

```bash
cp -r eval/robodojo /path/to/RoboDojo/XPolicyLab/policy/SANA_Robot
cd /path/to/RoboDojo/XPolicyLab/policy/SANA_Robot
bash install.sh sana-robot
```

The checkpoint comes with the description of its training data and the normalization of its robot rows:

```
checkpoints/sana_robot_ar_robodojo/
├── sana_robot_ar_robodojo.pth
├── meta.json
└── normalization.json
```

The adapter reads the policy's arguments and the camera sizes from [robodojo_ar.yaml](../../configs/robodojo_ar.yaml)
of the SANA repository the package is installed from, and the video VAE and the text encoder from its
`checkpoints/`; `SANA_ROBOT_VAE` and `SANA_ROBOT_TEXT_ENCODER` point at other places.

## 2. Run

```bash
cd /path/to/RoboDojo/XPolicyLab/policy/SANA_Robot
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
    <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

bash eval.sh RoboDojo stack_bowls /path/to/checkpoints/sana_robot_ar_robodojo arx_x5 ee 0 0 0 sana-robot <simulator env>
```

`eval.sh` starts the policy server and the simulator client; RoboDojo reports the task's score and success rate.
`EVAL_ENV_TYPE=debug` runs XPolicyLab's check of the adapter without the simulator. The simulator is Isaac Sim and
needs a GPU it supports; the policy server can run on another GPU or machine (see the deployment flow of
XPolicyLab).

## 3. Settings

[deploy.yml](deploy.yml) holds the inference settings of the released results:

| Key | Default | Meaning |
|---|---|---|
| `policy_type` | `ar` | `ar` for SANA-Robot-AR, `base` for SANA-Robot-base: the policy the checkpoint was trained as. |
| `steps` | `10` | Denoising steps per chunk. |
| `video_cfg_scale` | `6.0` | Guidance of the video of SANA-Robot-AR; `1.0` turns it off. |
| `n_action_steps` | `16` | Actions run of every chunk of 32 before the next one is generated; `null`: the whole chunk. |
| `async_delay_steps` | `0` | Actions the robot runs from the running chunk while the next one is generated. |
| `optimized`, `precision` | `false`, `bf16` | `true` runs the inference kernels and CUDA graphs of the main README at `precision`; `false` is the plain PyTorch path, on which the released results were measured. |
| `checkpoint_path`, `checkpoint_file`, `checkpoint_step` | `null` | The checkpoint directory in place of `ckpt_name`, the weights file in it, or the training step of a run. |
| `normalization_path`, `dataset_meta_path`, `vae_path`, `text_encoder_path` | `null` | Replace the files of the checkpoint directory and the two models. |
| `policy`, `view_sizes`, `chunk_size`, `video_frame_stride` | `null` | Replace what the training config says. |
| `cameras` | `null` | The observation's camera for every view of the dataset; `null`: `cam_head`, `cam_left_wrist`, `cam_right_wrist`. |
| `robot_root_poses` | `null` | The base pose of each arm for `action_type=ee`; `null`: RoboDojo's ARX-X5. |

The policy reads every fourth frame of the episode, so the client sends the observation after each action it runs,
as [deploy.py](deploy.py) does. At the start of an episode with `action_type=ee` the server prints how far the
end-effector pose the simulator reports is from the pose the policy reads; a gap of millimetres means that
`robot_root_poses` is not the simulator's.

## 4. Finetune

`process_data.sh` and `train.sh` run the converter and the trainer of the main README on RoboDojo's official
LeRobot v3.0 export, into `data/` and `checkpoints/` of the adapter:

```bash
bash process_data.sh RoboDojo cotrain arx_x5 ee
bash train.sh RoboDojo cotrain arx_x5 ee 0 0,1,2,3,4,5,6,7
bash eval.sh RoboDojo stack_bowls cotrain arx_x5 ee 0 0 0 sana-robot <simulator env>
```

`__init__.py`, `deploy.py`, `eval.sh`, `setup_eval_policy_server.sh` and `setup_eval_env_client.sh` are the files
of XPolicyLab's `policy/demo_policy`, unchanged.
