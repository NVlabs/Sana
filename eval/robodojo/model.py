# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""SANA-Robot in XPolicyLab: one ``sana_robot`` deployment, driven as the control loop of the environment client.

The policy code is the ``sana_robot`` package of the SANA repository. Its ``ChunkRunner`` takes the first frame and
state of an episode, then every ``video_frame_stride``-th frame, and hands out the action rows the robot runs; this
adapter feeds it the observations the client sends and returns its rows as XPolicyLab actions.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.policy.SANA_Robot import robodojo, runtime
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info

TAG = "[SANA_Robot]"
POLICY_DIR = Path(__file__).resolve().parent
POSE_CHECK_METRES = 0.005
POSE_CHECK_DEGREES = 1.0
SEED_STRIDE = 1_000_003


class Model(ModelTemplate):
    """SANA-Robot-AR or SANA-Robot-base served to one environment."""

    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = dict(model_cfg)
        self.action_type = str(model_cfg["action_type"])
        self.env_cfg_type = model_cfg["env_cfg_type"]
        self.seed = model_cfg.get("seed")

        if self.action_type not in robodojo.ACTION_TYPES:
            raise ValueError(f"action_type must be one of {robodojo.ACTION_TYPES}, got {self.action_type!r}")
        robodojo.check_robot(get_robot_action_dim_info(self.env_cfg_type))
        self.world_from_base = robodojo.root_transforms(model_cfg.get("robot_root_poses"))

        checkpoint = resolve_checkpoint_root(model_cfg, POLICY_DIR / "checkpoints", policy_dir=POLICY_DIR)
        run = runtime.resolve_run(model_cfg, checkpoint, runtime.sana_root(model_cfg))
        self._dataset = runtime.description_dataset(run.meta)
        self.deployment = runtime.build_deployment(model_cfg, run, self._dataset.name)
        views = [view.name for view in self.deployment.robot.info.views]
        self.cameras = list(model_cfg.get("cameras") or [robodojo.CAMERA_OF_VIEW[name] for name in views])
        if len(self.cameras) != len(views):
            raise ValueError(f"cameras must name one camera per view of the dataset {views}, got {self.cameras}")

        self._runner = None
        self._instruction = None
        self._episode = 0
        self.reset()
        config = self.deployment.config
        print(
            f"{TAG} ready: SANA-Robot-{config.policy_type} from {run.weights}, action_type={self.action_type}, "
            f"cameras={self.cameras}, steps={config.steps}, video_cfg_scale={config.video_cfg_scale}, "
            f"n_action_steps={config.n_action_steps}, async_delay_steps={config.async_delay_steps}, "
            f"optimized={config.optimized}, precision={config.precision}",
            flush=True,
        )

    def update_obs(self, obs):
        self._obs = obs
        if self._step is None:
            return
        self._step += 1
        if self._step % self._runner.controller.frame_stride == 0:
            self._runner.push_frame(*self._frame(obs))

    def update_obs_batch(self, obs_list):
        obs_list = [obs_list] if isinstance(obs_list, dict) else list(obs_list)
        if len(obs_list) != 1:
            raise ValueError(f"SANA_Robot serves one environment (eval_batch: false), got {len(obs_list)}")
        self.update_obs(obs_list[0])

    def _frame(self, obs) -> tuple:
        """Return what the runner reads of an observation: the frame of every camera and the robot row [80]."""
        images = robodojo.camera_images(obs["vision"], self.cameras)
        return images, robodojo.state_row(obs["state"])

    def get_action(self):
        if self._obs is None:
            raise RuntimeError("SANA_Robot needs an observation before get_action")
        if self._step is None:
            plan = self._open_episode(self._obs)
        else:
            controller = self._runner.controller
            due = self._plans * controller.n_action_steps + controller.async_delay_steps
            if self._step != due:
                raise RuntimeError(
                    f"SANA_Robot hands out the next actions after {due} executed actions of this episode, asked "
                    f"after {self._step}: run every action of a chunk and send the observation after each"
                )
            plan = self._runner.next_plan()
        self._plans += 1
        generate = [timing["ms"] for timing in self._runner.timings if timing["event"] == "generate"]
        self._runner.timings.clear()
        print(
            f"{TAG} episode {self._episode} chunk {plan.index} at step {self._step}: {len(plan.targets)} actions"
            + (f", generated in {generate[-1]:.0f} ms" if generate else ""),
            flush=True,
        )
        return robodojo.actions_from_rows(plan.targets, self.action_type, self.world_from_base)

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is not None and len(list(env_idx_list)) != 1:
            raise ValueError(f"SANA_Robot serves one environment (eval_batch: false), got {env_idx_list}")
        return [self.get_action()]

    def _open_episode(self, obs):
        """Start the runner on the first observation of an episode and return the plan of its first chunk."""
        instruction = obs.get("instruction")
        if instruction is None and obs.get("instructions"):
            instruction = obs["instructions"][0]
        instruction = str(instruction).strip() if instruction is not None else ""
        if self._runner is None or instruction != self._instruction:
            self.close()
            self._runner = self.deployment.runner(instruction)
            self._instruction = instruction
            print(f"{TAG} instruction: {instruction!r}", flush=True)
        self._episode += 1
        if self.seed is not None:
            torch.manual_seed(SEED_STRIDE * int(self.seed) + self._episode)
        images, row = self._frame(obs)
        if self.action_type == robodojo.ACTION_TYPE_EE:
            self._check_reported_pose(obs["state"], row)
        self._runner.timings.clear()
        self._step = 0
        return self._runner.start(images, row)

    def _check_reported_pose(self, state, row) -> None:
        """Compare the end-effector pose the environment reports with the one the policy reads, once per episode: they differ when the arms' root poses or the flange axes are not the simulator's, and then the pose targets are wrong too."""
        errors = robodojo.flange_pose_errors(state, row, self.world_from_base)
        if not errors:
            return
        text = ", ".join(
            f"{side} {1000 * metres:.2f} mm / {degrees:.2f} deg" for side, (metres, degrees) in errors.items()
        )
        print(f"{TAG} episode {self._episode}: reported end-effector pose against the joints' pose: {text}", flush=True)
        if any(metres > POSE_CHECK_METRES or degrees > POSE_CHECK_DEGREES for metres, degrees in errors.values()):
            warnings.warn(
                f"{TAG} the reported end-effector pose disagrees with the joints' pose ({text}); check robot_root_poses"
            )

    def reset(self):
        """Forget the episode; the next observation starts a new one."""
        self._obs = None
        self._step = None
        self._plans = 0

    def close(self):
        """Stop the runner's worker."""
        if self._runner is not None:
            self._runner.finish()
            self._runner = None
