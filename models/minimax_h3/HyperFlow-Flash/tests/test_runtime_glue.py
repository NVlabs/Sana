"""CPU checks of the glue (no diffusers / Triton / GPU / weights): sampler, schedule, presets, layout guards, sinks, SP."""
import types

import pytest
import torch

from sol_hyperflow_flash import config as C
from sol_hyperflow_flash import sampler as SM
from sol_hyperflow_flash import sp as SPM
from sol_hyperflow_flash.attention import (Controller, check_layout, dense_fp32, routable_blocks, target_video_start,
                                           text_audio_sink)

LIVE_T2V = [i for i in range(50) if i not in (6, 21, 22)]      # fl2va partition: dropped_blocks [6, 21, 22]
LIVE_REF = [i for i in range(50) if i != 9]                    # ref2va partition: dropped_blocks [9]


class Wrapper:
    def get_block_state(self, state):
        return state

    def set_block_state(self, state, bs):
        assert bs.noise_pred is not None and bs.audio_noise_pred is not None


def _denoise_module():
    mod = types.ModuleType("fake_denoise")
    mod.MiniMaxH3DenoiseLoopWrapper = Wrapper
    return mod


def test_sampler_six_steps_two_time_pairs():
    gen = torch.Generator().manual_seed(0)
    ctl = Controller(torch)
    ctl.sa = types.SimpleNamespace(_target_video_start=lambda vi, S: int(vi[0]))
    info = SM.install(ctl, torch, denoise_module=_denoise_module())
    sv, sa = info["video_sigmas"], info["audio_sigmas"]
    assert info["raw_sigmas"] == [1.0, 0.825217, 0.471725, 0.147401, 0.049626, 0.015517, 0.0]
    assert abs(sv[1] - 12 * .825217 / (1 + 11 * .825217)) < 1e-12 and sv[0] == 1.0 and sv[-1] == 0.0
    S, nv, na = 20, 10, 6
    text_i, audio_i, video_i = torch.arange(0, 4), torch.arange(4, 10), torch.arange(10, 20)
    tv, ta = torch.randn(nv, 3, generator=gen), torch.randn(na, 3, generator=gen)
    seen = []

    def fake_tr(**kw):
        seen.append(kw)
        t = kw["timestep"]
        sv_i, sa_i = 1 - float(t[0]), 1 - float(t[1])
        x, xa = kw["hidden_states"][0].float(), kw["audio_hidden_states"][0].float()
        return ((tv - x) / sv_i)[None], ((ta - xa) / sa_i)[None]

    bs = types.SimpleNamespace(latents=torch.randn(nv, 3, generator=gen), audio_latents=torch.randn(na, 3, generator=gen),
                               prompt_embeds=None, position_ids=torch.zeros(S, 3), token_tags=torch.zeros(S, dtype=torch.long),
                               video_indices=video_i, audio_indices=audio_i, text_indices=text_i, attention_kwargs=None,
                               num_condition_video_rows=0, num_condition_audio_rows=0, noise_pred=None, audio_noise_pred=None)
    ctl.preset = C.get("official")
    Wrapper().__call__(types.SimpleNamespace(transformer=fake_tr), bs)
    assert len(seen) == 6
    k0 = seen[0]
    assert torch.allclose(k0["timestep"], torch.tensor([0.0, 0.0, 0.999, 1.0]))
    assert abs(float(k0["endpoint_timestep"][0]) - (1 - sv[1])) < 1e-6
    assert abs(float(k0["endpoint_timestep"][1]) - (1 - sa[1])) < 1e-6
    idx = k0["timestep_indices"]
    assert (idx[text_i] == 0).all() and (idx[video_i] == 0).all() and (idx[audio_i] == 1).all()
    assert torch.allclose(bs.latents, tv, atol=1e-5) and torch.allclose(bs.audio_latents, ta, atol=1e-5)
    assert ctl.video_start == 10


def test_ref2va_sampler_reference_groups_and_layout():
    gen = torch.Generator().manual_seed(2)
    # packed [text 0-3 | ref video 4-7 | ref audio 8-9 | target audio 10-13 | target video 14-23]
    S = 24
    t8, vi8 = torch.arange(0, 4), torch.cat([torch.arange(4, 8), torch.arange(14, 24)])
    ai8 = torch.cat([torch.arange(8, 10), torch.arange(10, 14)])
    ncv, nca = 4, 2
    ctl = Controller(torch)
    info = SM.install(ctl, torch, denoise_module=_denoise_module(), component="transformer_ref")
    sv = info["video_sigmas"]
    ctl.preset = C.get("official", task="ref2va")
    lat, alat = torch.randn(14, 3, generator=gen), torch.randn(6, 3, generator=gen)
    cond, acond = lat[:ncv].clone(), alat[:nca].clone()
    seen = []

    def fake_tr(**kw):
        seen.append(kw)
        return torch.zeros(1, 14, 3), torch.zeros(1, 6, 3)

    bs = types.SimpleNamespace(latents=lat, audio_latents=alat, prompt_embeds=None, position_ids=torch.zeros(S, 3),
                               token_tags=torch.zeros(S, dtype=torch.long), video_indices=vi8, audio_indices=ai8,
                               text_indices=t8, attention_kwargs=None, num_condition_video_rows=ncv,
                               num_condition_audio_rows=nca, noise_pred=None, audio_noise_pred=None)
    Wrapper().__call__(types.SimpleNamespace(transformer_ref=fake_tr, transformer=None), bs)
    assert len(seen) == 6
    for i, kw in enumerate(seen):
        t, r, g = kw["timestep"], kw["endpoint_timestep"], kw["timestep_indices"]
        assert (g[t8] == 0).all() and (g[vi8[ncv:]] == 0).all() and (g[ai8[nca:]] == 1).all()
        assert (g[vi8[:ncv]] == 2).all() and (g[ai8[:nca]] == 3).all()
        assert abs(float(t[2]) - max(1 - sv[i], 0.999)) < 1e-6 and float(t[2]) == float(r[2])
        assert float(t[3]) == 1.0 and float(r[3]) == 1.0
    assert torch.equal(bs.latents[:ncv], cond) and torch.equal(bs.audio_latents[:nca], acond)
    assert ctl.video_start == 14 and ctl.layout["video_start"] == 14 and ctl.layout["n_target_video"] == 10
    bad_vi = torch.cat([torch.arange(4, 8), torch.arange(14, 24)])
    bad_ai = torch.cat([torch.arange(24, 26), torch.arange(10, 14)])
    with pytest.raises(RuntimeError):
        check_layout(torch, bad_vi, bad_ai, t8, 4, 2, 26, target_video_start(bad_vi, 26))
    assert target_video_start(torch.cat([torch.arange(5, 9), torch.arange(30, 40)]), 40) == 30


@pytest.mark.parametrize("task,live,expected", [
    ("t2va", LIVE_T2V, {"dense": 110, "solbsa": 172}),
    ("ref2va", LIVE_REF, {"dense": 114, "solbsa": 180}),
])
def test_schedule_counts(task, live, expected):
    ctl = Controller(torch)
    for name, exp in (("official", {"dense": 6 * len(live)}), ("sol-engine", expected)):
        ctl.preset = C.get(name, task=task)
        cnt = {}
        for step in range(6):
            ctl.step = step
            for b in live:
                cnt[ctl.backend(b)] = cnt.get(ctl.backend(b), 0) + 1
        assert cnt == exp, (task, name, cnt)
    ctl.preset = C.get("sol-engine", task=task, attention_backend="dense")
    assert {ctl.backend(b) for b in live} == {"dense"}


def test_presets():
    t, r = C.get("sol-engine"), C.get("sol-engine", task="ref2va")
    assert (t.tau, t.dense_steps, t.edge_blocks, t.fusions, t.merge_tuned, t.vae, t.te_trim, t.transport, t.sink) == \
        (1.25, 2, (0, 1, 48, 49), True, True, "solh3", True, "int8fp8-sparse", "prefix")
    assert (r.sink, r.ref_encode, r.encode_compile, r.prep_threads) == ("text_audio", "global", True, 32)
    assert r.as_dict() | dict(sink="prefix", ref_encode="diffusers", encode_compile=False, prep_threads=0, notes="") == \
        t.as_dict() | dict(notes="")
    s = C.get("official")
    assert not (s.sparse or s.fusions or s.te_trim) and s.vae == "diffusers" and s.transport == "bf16"
    d = C.get("sol-engine", attention_backend="dense")
    assert not d.sparse and d.fusions and d.name == "sol-engine-dense"
    assert C.get("sol-engine", tau=1.5).tau == 1.5
    with pytest.raises(KeyError):
        C.get("unknown")
    for gone in ("stock", "ours"):
        with pytest.raises(KeyError):
            C.get(gone)


def test_text_audio_sink_and_routable_blocks():
    # text 0-99 | ref video 100-299 | ref audio 300-339 | target audio 340-379 | target video 380-639 (S 640, 10 blocks)
    t9, ai9 = torch.arange(0, 100), torch.cat([torch.arange(300, 340), torch.arange(340, 380)])
    idx, mask = text_audio_sink(torch, t9, ai9, 640)
    assert idx.numel() == 180 and mask.tolist() == [True, True, False, False, True, True, False, False, False, False]
    assert routable_blocks(torch, 10, 380, mask, "cpu").tolist() == [2, 3, 6, 7, 8, 9]
    assert routable_blocks(torch, 10, 380, None, "cpu").tolist() == [6, 7, 8, 9]
    ctl = Controller(torch)
    ctl.preset = C.get("sol-engine", task="ref2va")
    ctl.video_start = 380
    ctl.on_layout(torch.cat([torch.arange(100, 300), torch.arange(380, 640)]), ai9, t9, 200, 40, 640)
    assert ctl.layout["sink"] == "text_audio" and ctl.layout["sink_rows"] == 180
    assert torch.equal(ctl.sink_block_mask, mask) and ctl.visual_ref_idx.numel() == 200


def test_dense_fp32_reference():
    gen = torch.Generator().manual_seed(3)
    q, k, v = (torch.randn(1, 37, 3, 16, generator=gen) for _ in range(3))
    ref = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                                           v.transpose(1, 2)).transpose(1, 2)
    assert torch.allclose(dense_fp32(torch, q, k, v, chunk=8), ref, atol=1e-5)


@pytest.mark.parametrize("world,heads", [(4, 8), (8, 56)])
def test_sequence_parallel_data_movement(world, heads):
    gen = torch.Generator().manual_seed(4)
    S, E = 37, 6
    full = torch.randn(1, S, heads, E, generator=gen)
    noop = lambda *a: None  # noqa: E731
    ranks = [SPM.SP(world, r, a2a=noop, allgather=noop) for r in range(world)]
    shards = [ranks[r].shard(full, S, 1) for r in range(world)]
    assert sum(s.shape[1] for s in shards) == S and torch.equal(torch.cat(shards, 1), full)
    sends = [ranks[r].send_seq_to_head(shards[r]) for r in range(world)]
    hd = [SPM.SP.assemble_seq_to_head([sends[j][r] for j in range(world)]) for r in range(world)]
    hw = heads // world
    for r in range(world):
        assert torch.equal(hd[r], full[:, :, r * hw:(r + 1) * hw])
    outs = [h * 2 + 1 for h in hd]
    sends2 = [ranks[r].send_head_to_seq(outs[r], S) for r in range(world)]
    back = [SPM.SP.assemble_head_to_seq([sends2[j][r] for j in range(world)]) for r in range(world)]
    assert torch.equal(torch.cat(back, 1), full * 2 + 1)
    c = max(ranks[0].sizes(S))
    padded = [torch.nn.functional.pad(s[0, :, 0], (0, 0, 0, c - s.shape[1])) for s in shards]
    g = SPM.SP(world, 1, a2a=noop, allgather=lambda out, inp: out.copy_(torch.cat(padded, 0))).gather(shards[1][:, :, 0], S)
    assert torch.equal(g, full[:, :, 0])


def test_seed_precedence():
    from sol_hyperflow_flash.references import resolve_seed

    assert resolve_seed(7, {"seed": 3}) == 7
    assert resolve_seed(None, {"seed": 3}) == 3
    assert resolve_seed(None, {"seed": None}) == 0 and resolve_seed(None) == 0
    assert resolve_seed(0, {"seed": 3}) == 0


def test_per_rank_cache_defaults(tmp_path, monkeypatch):
    import tempfile

    from sol_hyperflow_flash import runtime as RT

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    for var in (*RT.COMPILE_CACHES, "CUDA_CACHE_PATH", "XDG_CACHE_HOME", "TMPDIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    import os

    for run_id, port, key in (("3f2a-9c", "29500", "_3f2a-9c"), ("none", "29511", "_port29511")):
        for var in RT.COMPILE_CACHES:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("TORCHELASTIC_RUN_ID", run_id)
        monkeypatch.setenv("MASTER_PORT", port)
        RT.per_rank_caches()
        for var in RT.COMPILE_CACHES:
            path = os.environ[var]
            assert path.startswith(str(tmp_path)) and path.endswith("_rank1") and os.path.isdir(path), (var, path)
            assert key + os.sep in path, (key, path)


def test_fused_projections_rejected():
    from sol_hyperflow_flash.attention import make_processor

    proc = make_processor(Controller(torch), 0, None)
    attn = types.SimpleNamespace(fused_projections=True)
    with pytest.raises(RuntimeError, match="fused QKV"):
        proc(attn, torch.zeros(1, 4, 8))
