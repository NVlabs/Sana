"""The HyperFlow H3 Flash 6-step two-time sampler (model card), as the denoise loop of the H3 modular pipeline.

raw sigmas RAW = [1, .825217, .471725, .147401, .049626, .015517, 0]; shift(s, k) = k s / (1 + (k - 1) s), k = 12 video, k = 3 audio.
Step i: generated video + text rows (t, r) = (1 - sv_i, 1 - sv_{i+1}); generated audio rows (1 - sa_i, 1 - sa_{i+1});
conditioning video rows - fl2va keyframes, ref2va image / video references - (max(1 - sv_i, .999), same); reference audio rows
(1, 1). The same table for both partitions (model card "Sampling"); ref2va runs it on the pipeline's `transformer_ref`.
v = model(...); x0 = x + s_i v; x <- (s_{i+1} / s_i) x + (1 - s_{i+1} / s_i) x0 on the generated rows only (fp32 math); no CFG.

Pairs are passed per row GROUP (the base pipeline's build_row_timesteps uniques by VALUE, which would merge video and audio at
step 0 where both start at t = 0 but end at different r) - why the base loop is not a drop-in sampler.
Everything else (text encoding, layout, latents, video / audio decode) is the base MiniMax-H3 modular pipeline, unchanged.
"""
from __future__ import annotations

RAW_SIGMAS = [1.0, 0.825217, 0.471725, 0.147401, 0.049626, 0.015517, 0.0]
VIDEO_SHIFT, AUDIO_SHIFT, KEYFRAME_T = 12.0, 3.0, 0.999
N_STEPS = len(RAW_SIGMAS) - 1


def shift(s: float, k: float) -> float:
    return k * s / (1.0 + (k - 1.0) * s)


def sigma_grids():
    return [shift(s, VIDEO_SHIFT) for s in RAW_SIGMAS], [shift(s, AUDIO_SHIFT) for s in RAW_SIGMAS]


def install(ctl, torch, denoise_module=None, component="transformer"):
    """Replace MiniMaxH3DenoiseLoopWrapper.__call__ (t2va / fl2va and ref2va denoise steps subclass it) with the 6-step sampler
    driving `components.<component>` ("transformer", or "transformer_ref" for ref2va). `ctl` hooks (all optional):
    ctl.on_request(video_indices, S) and ctl.on_layout(...) once per request, ctl.on_step(i) before each forward,
    ctl.after_step(i, bs) after the update."""
    if denoise_module is None:
        from diffusers.modular_pipelines.minimax_h3 import denoise as denoise_module
    sv, sa = sigma_grids()

    @torch.no_grad()
    def loop(self, components, state):
        bs = self.get_block_state(state)
        tr = getattr(components, component)
        dev = bs.latents.device
        ncv, nca = int(bs.num_condition_video_rows or 0), int(bs.num_condition_audio_rows or 0)
        S = int(bs.position_ids.shape[0])
        vi, ai = bs.video_indices, bs.audio_indices
        grp = torch.zeros(S, dtype=torch.long, device=dev)       # 0: generated video + text rows
        grp[ai[nca:]] = 1                                           # 1: generated audio rows
        if ncv:
            grp[vi[:ncv]] = 2                                       # 2: keyframe-conditioning video rows
        if nca:
            grp[ai[:nca]] = 3                                       # 3: reference audio rows
        if hasattr(ctl, "on_request"):
            ctl.on_request(vi, S)
        if hasattr(ctl, "on_layout"):
            ctl.on_layout(vi, ai, bs.text_indices, ncv, nca, S)
        for i in range(N_STEPS):
            tv, rv = 1.0 - sv[i], 1.0 - sv[i + 1]
            ta, ra = 1.0 - sa[i], 1.0 - sa[i + 1]
            tc = max(tv, KEYFRAME_T)
            t = torch.tensor([tv, ta, tc, 1.0], dtype=torch.float32, device=dev)
            r = torch.tensor([rv, ra, tc, 1.0], dtype=torch.float32, device=dev)
            if hasattr(ctl, "on_step"):
                ctl.on_step(i)
            vel_v, vel_a = tr(hidden_states=bs.latents[None], audio_hidden_states=bs.audio_latents[None],
                              encoder_hidden_states=bs.prompt_embeds, timestep=t, timestep_indices=grp,
                              token_tags=bs.token_tags, position_ids=bs.position_ids, video_indices=vi, audio_indices=ai,
                              text_indices=bs.text_indices, endpoint_timestep=r, attention_kwargs=bs.attention_kwargs,
                              return_dict=False)
            for lat, vel, n0, s0, s1 in ((bs.latents, vel_v, ncv, sv[i], sv[i + 1]),
                                         (bs.audio_latents, vel_a, nca, sa[i], sa[i + 1])):
                x = lat[n0:].float()
                x0 = x + s0 * vel[0, n0:].float()
                ratio = s1 / s0
                lat[n0:] = (ratio * x + (1.0 - ratio) * x0).to(lat.dtype)
            bs.noise_pred, bs.audio_noise_pred = vel_v, vel_a     # declared intermediate outputs of the loop block
            if hasattr(ctl, "after_step"):
                ctl.after_step(i, bs)
        self.set_block_state(state, bs)
        return components, state

    denoise_module.MiniMaxH3DenoiseLoopWrapper.__call__ = loop
    return dict(raw_sigmas=RAW_SIGMAS, video_sigmas=sv, audio_sigmas=sa, video_shift=VIDEO_SHIFT, audio_shift=AUDIO_SHIFT,
                keyframe_t=KEYFRAME_T)
