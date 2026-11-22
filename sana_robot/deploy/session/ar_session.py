# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sana_robot.deploy.memory.softmax_memory import SoftmaxMemory
from sana_robot.deploy.runtime.cuda_graph import CudaGraphDriver
from sana_robot.deploy.sampling.flow_euler import (
    draw_noise,
    euler_step,
    flow_schedule,
    host_geometry,
    sample_dtype,
    session_verb,
)
from sana_robot.policy.layout.multiview import batch_view_shapes
from sana_robot.policy.sana_robot_ar import SanaRobotAR

GRAPH_CLOCK_CHUNKS = 1024


class ARSession:
    """Streaming inference of SANA-Robot-AR over one episode.

    The session works on latents. ``start`` opens the memory with the sink, the latent of the episode's first
    frame. Every request then commits the chunk the robot has just executed (``commit``, from the second request
    on) and generates the next chunk from the current observation and state (``generate``). A softmax layer
    remembers the sink and the last N committed chunks, a Gated DeltaNet layer one state over all of them.

    With classifier-free guidance of the video the session keeps a second memory, the one the same episode leaves
    under the unconditional prompt: every forward runs once per prompt, and a denoising step moves the video along
    unconditional + scale * (conditional - unconditional) of the two video velocities. The actions follow the
    conditional forward alone.

    An optimized session replays a commit and a generation as CUDA graphs on buffers that keep their addresses,
    for episodes of up to ``GRAPH_CLOCK_CHUNKS`` chunks; past that it runs them eagerly.
    """

    def __init__(
        self,
        policy: SanaRobotAR,
        text_embeddings: torch.Tensor,
        text_mask: torch.Tensor,
        data_info: dict,
        steps: int,
        video_flow_shift: float = 5.0,
        action_flow_shift: float = 1.0,
        optimized: bool = True,
        video_cfg_scale: float = 1.0,
        unconditional_embeddings: torch.Tensor | None = None,
        unconditional_mask: torch.Tensor | None = None,
    ) -> None:
        """Build a session for one task prompt.

        Args:
            policy: The policy, in eval mode on its device.
            text_embeddings: Text embeddings of the prompt groups [1, V + 1, 1, L_text, C_text].
            text_mask: Text token mask [1, V + 1, L_text].
            data_info: What stays the same over the episode: ``view_latent_shape``, ``view_slot_ids``,
                ``model_fps`` and ``video_frame_stride``.
            steps: Denoising steps per chunk.
            video_flow_shift: Flow shift of the video latents.
            action_flow_shift: Flow shift of the actions.
            optimized: Replay commits and generations as CUDA graphs; needs a CUDA device.
            video_cfg_scale: Classifier-free guidance scale of the video velocity; 1 turns the guidance off.
            unconditional_embeddings: Text embeddings of the prompt groups without the instruction, as
                ``text_embeddings``; read with a guidance scale other than 1.
            unconditional_mask: Their text token mask.
        """
        self.policy = policy
        self.video_cfg_scale = float(video_cfg_scale)
        self.texts = [(text_embeddings.to(policy.dtype), text_mask)]
        if self.video_cfg_scale != 1.0:
            self.texts.append((unconditional_embeddings.to(policy.dtype), unconditional_mask))
        self.data_info = host_geometry(data_info)
        self.steps = int(steps)
        device = next(policy.parameters()).device
        (self.video_timesteps, self.video_levels) = flow_schedule(self.steps, video_flow_shift, device)
        (self.action_timesteps, self.action_levels) = flow_schedule(self.steps, action_flow_shift, device)
        (self.chunk_actions, self.chunk_frames) = policy.chunk_size(self.data_info)
        view_areas = [height * width for (height, width) in batch_view_shapes(self.data_info)]
        self.memory = SoftmaxMemory(
            view_areas, self.chunk_frames, self.chunk_actions, policy.sliding_window_chunks, policy.vae_temporal_stride
        )
        self.softmax_stores = [
            {layer: self.memory.add_layer() for layer in policy.softmax_layer_indices} for _ in self.texts
        ]
        self.states = []
        self.optimized = optimized and device.type == "cuda"
        (self.commit_step, self.generate_step) = (self._commit, self._generate)
        self.packed_text = {} if self.optimized else None
        if self.optimized:
            self.memory.clock_floor = GRAPH_CLOCK_CHUNKS
            (self.commit_step, self.generate_step) = (CudaGraphDriver(self._commit), CudaGraphDriver(self._generate))

    @property
    def chunk(self) -> int:
        """Index of the chunk the next ``generate`` produces: the number of committed chunks."""
        return self.memory.count

    def set_prompt(self, text_embeddings: torch.Tensor, text_mask: torch.Tensor) -> None:
        """Change the task prompt of the running episode; the memory stays as the earlier prompt left it."""
        self.texts[0] = (text_embeddings.to(self.policy.dtype), text_mask)
        if self.optimized:
            (self.commit_step, self.generate_step) = (CudaGraphDriver(self._commit), CudaGraphDriver(self._generate))

    @session_verb
    def start(self, sink_latent: torch.Tensor) -> None:
        """Open the memory of a new episode with the sink.

        Args:
            sink_latent: The first frame of every view encoded alone, [1, C, 1, 1, sum(H_v * W_v)].
        """
        self.memory.reset()
        text_caches = self._text_caches(None)
        for (text_embeddings, text_mask), caches in zip(self.texts, text_caches):
            self.policy.forward_sink(
                sink_latent,
                sink_latent.new_zeros(1, 1, 1),
                text_embeddings,
                text_mask,
                kv_caches=caches,
                data_info=self.data_info,
            )
        self.memory.advance()
        self._keep_states(self._new_states(text_caches))

    @session_verb
    def commit(self, chunk_latents: torch.Tensor) -> None:
        """Write the chunk the robot has executed into the memory.

        Args:
            chunk_latents: The clean target latents of the chunk, the frames observed while it ran,
                [1, C, f, 1, sum(H_v * W_v)].
        """
        self.memory.refresh_phases()
        step = self.commit_step if self._graph_ready() else self._commit
        states = step(chunk_latents.float(), self._chunk_index(chunk_latents.device), self.memory.kept, *self.states)
        self._keep_states(states)
        self.memory.advance()

    @session_verb
    def generate(
        self,
        observation_latent: torch.Tensor,
        state: torch.Tensor,
        state_mask: torch.Tensor,
        action_mask: torch.Tensor,
        video_noise: torch.Tensor | None = None,
        action_noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate the next chunk from the current observation and state. The memory does not change.

        Args:
            observation_latent: The current frame of every view encoded alone, [1, C, 1, 1, sum(H_v * W_v)]; for
                the first chunk the sink latent.
            state: The normalized robot state [1, 80].
            state_mask: Which state dimensions the robot has, [1, 80].
            action_mask: Which action dimensions the robot has, [1, A, 80].
            video_noise: Starting noise of the target latents [1, C, f, 1, sum(H_v * W_v)]; drawn when None.
            action_noise: Starting noise of the actions [1, A, 80]; drawn when None.
            generator: Random generator of the noise that is drawn.

        Returns:
            The normalized actions [1, A, 80] and the predicted target latents [1, C, f, 1, sum(H_v * W_v)].
        """
        (device, dtype) = (observation_latent.device, sample_dtype(self.policy))
        if video_noise is None:
            shape = (1, observation_latent.shape[1], self.chunk_frames, *observation_latent.shape[3:])
            video_noise = draw_noise(shape, generator, device, dtype)
        if action_noise is None:
            action_noise = draw_noise(tuple(action_mask.shape), generator, device, dtype)
        self.memory.refresh_phases()
        step = self.generate_step if self._graph_ready() else self._generate
        (action, video) = step(
            observation_latent.to(dtype),
            state.to(dtype),
            state_mask,
            action_mask,
            video_noise.to(dtype),
            action_noise.to(dtype),
            self._chunk_index(device),
            self.memory.kept,
            *self.states,
        )
        return (action.clone(), video.clone())

    @session_verb
    def save_memory(self) -> dict:
        """Return a copy of the memory: the softmax stores and every Gated DeltaNet state."""
        return {"softmax": self.memory.save(), "states": [state.clone() for state in self.states]}

    @session_verb
    def load_memory(self, memory: dict) -> None:
        """Bring a saved memory back into the session's own buffers."""
        self.memory.load(memory["softmax"])
        self._keep_states(memory["states"])

    def _graph_ready(self) -> bool:
        """Return whether the chunk at hand can replay as a graph: an optimized session inside the clock the graphs were captured for."""
        return self.optimized and self.memory.count < self.memory.clock_floor

    def _chunk_index(self, device) -> torch.Tensor:
        """Return the index of the chunk at hand as a device scalar, an input of the graphs."""
        return torch.tensor(self.chunk, dtype=torch.long, device=device)

    def _text_caches(self, states) -> list:
        """Return, per prompt, the cache of every layer for one forward: a softmax layer's store, a Gated DeltaNet layer's state. ``states`` holds the states of the first prompt, then those of the second."""
        states = list(states or ())
        count = len(states) // len(self.texts)
        text_caches = []
        for index, stores in enumerate(self.softmax_stores):
            own = iter(states[index * count : (index + 1) * count])
            caches = []
            for layer in range(len(self.policy.blocks)):
                if layer in stores:
                    caches.append({"kv": stores[layer]})
                else:
                    state = next(own, None)
                    caches.append({} if state is None else {"S": state})
            text_caches.append(caches)
        return text_caches

    @staticmethod
    def _new_states(text_caches) -> tuple:
        """Return the Gated DeltaNet states the forwards left in the caches, prompt after prompt."""
        return tuple(cache["S"] for caches in text_caches for cache in caches if "S" in cache)

    def _keep_states(self, states) -> None:
        """Copy new Gated DeltaNet states into the session's state buffers, which keep their addresses."""
        if not self.states:
            self.states = [state.clone() for state in states]
            return
        for buffer, state in zip(self.states, states):
            buffer.copy_(state)

    def _commit(self, chunk_latents, chunk, kept, *states) -> tuple:
        """Run the commit forward of a clean chunk on the given states and return the new states; the softmax stores take the chunk in place."""
        text_caches = self._text_caches(states)
        for (text_embeddings, text_mask), caches in zip(self.texts, text_caches):
            self.policy.forward_chunk(
                chunk_latents,
                chunk_latents.new_zeros(1, 1, self.chunk_frames),
                text_embeddings,
                text_mask,
                kv_caches=caches,
                save_kv_cache=True,
                chunk_offset_tensor=chunk,
                data_info=self.data_info,
            )
        return self._new_states(text_caches)

    def _generate(self, observation, state, state_mask, action_mask, video, action, chunk, kept, *states) -> tuple:
        """Read the observation, then denoise the chunk from the given noise; returns the actions and the target latents."""
        text_caches = self._text_caches(states)
        action = action.masked_fill(~action_mask, 0)
        observation_info = dict(self.data_info, initial_state=state, initial_state_condition_mask=state_mask)
        for (text_embeddings, text_mask), caches in zip(self.texts, text_caches):
            self.policy.forward_observation(
                observation,
                observation.new_zeros(1, 1, 1),
                text_embeddings,
                text_mask,
                kv_caches=caches,
                chunk_offset_tensor=chunk,
                data_info=observation_info,
            )
        for step in range(self.steps):
            step_info = dict(
                self.data_info,
                action=action,
                action_mask=action_mask,
                action_timestep=self.action_timesteps[step].expand(1, self.chunk_actions),
            )
            predictions = [
                self.policy.forward_chunk(
                    video,
                    self.video_timesteps[step].expand(1, 1, self.chunk_frames),
                    text_embeddings,
                    text_mask,
                    kv_caches=caches,
                    save_kv_cache=False,
                    chunk_offset_tensor=chunk,
                    data_info=step_info,
                )
                for (text_embeddings, text_mask), caches in zip(self.texts, text_caches)
            ]
            video_velocity = predictions[0]["video_pred"]
            if len(predictions) > 1:
                unconditional = predictions[1]["video_pred"]
                video_velocity = unconditional + self.video_cfg_scale * (video_velocity - unconditional)
            video = euler_step(video, video_velocity, self.video_timesteps[step], self.video_levels[step + 1])
            action = euler_step(
                action, predictions[0]["action_pred"], self.action_timesteps[step], self.action_levels[step + 1]
            )
            action = action.masked_fill(~action_mask, 0)
        return (action, video)
