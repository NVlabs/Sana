"""End-to-end runtime: the MiniMax-H3 modular pipeline with the HyperFlow H3 Flash transformer and its 6-step sampler,
plus the Sol-Engine optimisations of `config` switched per preset.

    rt = HyperFlowFlashRuntime(weights, base_model)               # torchrun with world > 1: Ulysses sequence parallelism
    rt.apply(config.get("sol-engine"))
    rec, media = rt.generate(prompt, num_frames=124, seed=0, keep=True)

    rt = HyperFlowFlashRuntime(weights, base_model, task="ref2va")      # the ref2va partition as `transformer_ref`
    rec, media = rt.generate(prompt, num_frames=124, seed=0, references=[MiniMaxH3Reference(...), ...])

t2va: the H3 `t2va` modular pipeline; ref2va: diffusers MiniMaxH3Ref2VABlocks (setup, Qwen3-VL text encoder over the prompt and
the image / video references, reference VAE encoders, packed layout, latents, denoise, decode). Text encoder, layout, latents (CPU
generator), video VAE and audio VAE are the base pipeline's. Under sequence parallelism the live DiT blocks are parallel (`sp`);
presets with `vae="solh3"` run Sol-H3's parallel compiled video VAE decode and presets with `ref_encode` != "stock" shard the
reference encode (`vae_par`); `prep_threads` resizes the reference video frames on a host thread pool (same PIL calls, bitwise).
"""
from __future__ import annotations

import os
import time

from . import checkpoint as CK
from . import config as C
from . import sampler as SM
from .attention import Controller, make_processor
from .media import event

FPS = 24
HEIGHT, WIDTH = 768, 1344
DURATION_FRAMES = {5: 124, 10: 243, 15: 362}     # the model's 17n + 5 frame rule (Sol-H3 engine presets)


def _get(state, name):
    v = state.get(name) if hasattr(state, "get") else None
    return v if v is not None else getattr(state, name, None)


def dist_env():
    return int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


TASKS = {"t2va": "transformer", "ref2va": "transformer_ref"}     # task -> pipeline slot the HyperFlow transformer replaces


class HyperFlowFlashRuntime:
    def __init__(self, weights=CK.HF_REPO, base_model="MiniMaxAI/MiniMax-H3", sol_h3_dir=None, *, task="t2va",
                 revision=CK.HF_REVISION, sol_bucket=4096, te_offload=False, need=("sparse", "fusions")):
        import torch

        if task not in TASKS:
            raise ValueError(f"task {task!r} not in {sorted(TASKS)}")
        self.task, self.component = task, TASKS[task]
        self.torch = torch
        self.rank, self.local_rank, self.world = dist_env()
        self.model_dir = CK.resolve(weights, task, revision=revision)
        self.h3_dir = base_model
        self.sol_h3_dir = sol_h3_dir
        self.te_offload = te_offload
        self.pvae = None              # vae_par.ParallelVAE, created by the first preset that needs it
        self.prep_threads = 0         # host threads for the reference-frame resize of the ref2va setup step (0 = stock)
        self.audit = False            # stage audit: host-synced per-block wall times + SP collective times, gathered per rank
        torch.backends.cuda.matmul.allow_tf32 = False
        self.dev = torch.device("cuda", self.local_rank)
        torch.cuda.set_device(self.dev)
        if self.world > 1:
            import torch.distributed as dist

            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", device_id=self.dev)
        self.ctl = Controller(torch, sol_bucket=sol_bucket)
        if need:
            from . import solh3

            mods = solh3.load(sol_h3_dir, modules=("sparse_attention", "fusions", "fusion_install", "sol_residual"))
            self.ctl.sa, self.ctl.fusions, self.fusion_install = mods.sparse_attention, mods.fusions, mods.fusion_install
            from . import merge_tuned

            self.merge_ref = mods.sparse_attention.merge_sol_residual
            self.merge_tuned = merge_tuned.make(mods.sol_residual)
            self.ctl.merge_ref = self.merge_ref
        else:
            self.fusion_install = None

        t0 = time.perf_counter()
        from diffusers import ComponentsManager, ModularPipeline
        from diffusers.modular_pipelines.minimax_h3 import before_encoder

        before_encoder.MINIMAX_H3_MAX_DURATION = 364 / FPS      # accept the 362-frame (15 s) preset, as the Sol-H3 engine does
        if task == "ref2va":
            from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Ref2VABlocks

            pipe = MiniMaxH3Ref2VABlocks().init_pipeline(str(self.h3_dir), components_manager=ComponentsManager())
        else:
            pipe = ModularPipeline.from_pretrained(str(self.h3_dir), components_manager=ComponentsManager())
        names = [n for n, s in pipe._component_specs.items()
                 if s.default_creation_method == "from_pretrained" and n != self.component]
        pipe.load_components(names=names, dtype=torch.bfloat16, pretrained_model_name_or_path=str(self.h3_dir))
        self.ckpt = CK.load_transformer(self.model_dir, torch)
        tr = self.ckpt.model
        pipe.update_components(**{self.component: tr})
        pipe.to(self.dev)
        if te_offload:
            pipe.text_encoder.to("cpu")
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.pipe, self.tr = pipe, tr
        self.sampler_info = SM.install(self.ctl, torch, component=self.component)
        self.live = [(i, b) for i, b in enumerate(tr.transformer_blocks) if not isinstance(b, self.ckpt.dropped_cls)]
        self.load_s = time.perf_counter() - t0

        from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb

        self.stock_procs = {i: b.attn.processor for i, b in self.live}
        if self.world > 1:
            from . import sp as SPM

            self.ctl.sp = SPM.SP(self.world, self.rank)
            SPM.install_block_hooks([b for _, b in self.live], self.ctl.sp)
        self.side_procs = {i: make_processor(self.ctl, i, _apply_rotary_emb) for i, _ in self.live}
        self._fusion_restores = []
        self._install_timers()

    # ---- presets ---------------------------------------------------------------------------------------------------------------
    def int8_path(self):
        """INT8 GEMM the checkpoint code runs on this GPU ("cublas" / "triton" / ...)."""
        return self.ckpt.int8.resolve_int8_gemm(self.dev)

    def apply(self, preset: C.Preset):
        ctl = self.ctl
        if ctl.sa is not None:
            ctl.sa.merge_sol_residual = self.merge_tuned if preset.merge_tuned else self.merge_ref
        if (preset.sparse or preset.fusions) and ctl.sa is None:
            raise RuntimeError("preset needs the Sol-H3 runtime (--sol-h3-dir)")
        if ctl.preset is not None and ctl.preset.name != preset.name:
            # preset switch (never inside a timed request): release the cached blocks of the previous preset (e.g. the parallel
            # decoder's batched tiles) so they cannot slow the next preset's allocations
            self.torch.cuda.empty_cache()
        ctl.preset = preset
        if (preset.vae != "stock" or preset.ref_encode != "stock" or preset.encode_compile) and self.pvae is None:
            from . import solh3, vae_par

            vp = solh3.load(self.sol_h3_dir, modules=("vae_parallel",)).vae_parallel
            self.pvae = vae_par.ParallelVAE(self.torch, self.pipe.vae, vp)
        if self.pvae is not None:
            self.pvae.set(preset.vae, preset.ref_encode, preset.encode_compile)
        if preset.transport != "bf16" and self.world > 1 and ctl.q8 is None:
            from . import solh3

            ctl.q8 = solh3.load(self.sol_h3_dir, modules=("relayout", "comm_quant", "ulysses"))
        self.prep_threads = int(preset.prep_threads)
        self.set_te_trim(preset.te_trim)
        plain = not (preset.sparse or preset.fusions) and ctl.sp is None
        for i, b in self.live:
            b.attn.set_processor(self.stock_procs[i] if plain else self.side_procs[i])
        ctl.block_fused = True
        ctl.fused_qknorm_rope = preset.fusions
        if preset.fusions and not self._fusion_restores:
            class _Live:                     # fusion_install._patch_blocks iterates .transformer_blocks; hand it the live ones
                transformer_blocks = [b for _, b in self.live]

            self._fusion_restores.extend(self.fusion_install._patch_blocks(_Live, True, True, lora=False))
            for blk, _name, orig in self._fusion_restores:     # step-switchable: original forward while ctl.block_fused is False
                fused = blk.forward
                blk.forward = (lambda *a, _f=fused, _o=orig, **k: _f(*a, **k) if ctl.block_fused else _o(*a, **k))
        if not preset.fusions and self._fusion_restores:
            for module, name, original in self._fusion_restores:
                setattr(module, name, original)
            self._fusion_restores.clear()

    def set_te_trim(self, on):
        """Conditioner trimming (as ../HyperFlow/sol_hyperflow/conditioner_memory.py): MiniMax-H3 reads the pre-norm
        hidden_states[50] of its Qwen3-VL conditioner, so the language layers after layer 50 never contribute; keep 51 layers
        (hidden_states[50] stays pre-norm). Switchable per preset (the full layer list is kept); the output is bitwise the
        untrimmed one."""
        if self._te_layers is None:
            if not on:
                return
            import torch

            from diffusers.modular_pipelines.minimax_h3 import encoders as h3enc

            keep = int(h3enc.MINIMAX_H3_TEXT_ENCODER_LAYER) + 1
            lang = self.pipe.text_encoder.model.language_model
            self._te_layers = (lang.layers, torch.nn.ModuleList(list(lang.layers[:keep])), len(lang.layers), keep)
        full, trim, n_full, n_keep = self._te_layers
        te = self.pipe.text_encoder
        te.model.language_model.layers = trim if on else full
        n = n_keep if on else n_full
        te.model.language_model.config.num_hidden_layers = n
        te.config.text_config.num_hidden_layers = n

    def expected_counts(self, preset: C.Preset, steps=SM.N_STEPS):
        live = {i for i, _ in self.live}
        if not preset.sparse:
            return {"stock": steps * len(live)} if (self.ctl.sp is not None) else {}
        edge = len(set(preset.edge_blocks) & live)
        dense = min(preset.dense_steps, steps)
        return {"stock": dense * len(live) + (steps - dense) * edge, "solbsa": (steps - dense) * (len(live) - edge)}

    # ---- timers ----------------------------------------------------------------------------------------------------------------
    def _install_timers(self):
        torch = self.torch
        self._stage, self._fwd, self._starts, self._depth = {}, [], [], {}
        self._te_open = {}
        self._capture, self._captured = {"on": False}, {}
        self.captured_raw = {}           # decoder inputs of the last capture_latents request, original dtype (VAE gate)
        self.captured_er_args = None     # encode_references args of the last capture_latents request (encode gate)
        self.captured_te_args = None     # encode_prompt args of the last capture_latents request (text-encoder trim gate)
        self.captured_cond = None        # conditioning after the reference-encoder step of the last capture request
        self._te_layers = None           # (all layers, layers up to the selected hidden state) of the Qwen3-VL language model
        self.tr.register_forward_pre_hook(lambda m, a, kw: self._starts.append(event(torch)), with_kwargs=True)
        self.tr.register_forward_hook(lambda m, a, kw, out: self._fwd.append((self._starts.pop(), event(torch))), with_kwargs=True)
        from diffusers.modular_pipelines.minimax_h3 import encoders as h3enc

        te_cls = h3enc.MiniMaxH3Ref2VATextEncoderStep if self.task == "ref2va" else h3enc.MiniMaxH3TextEncoderStep
        orig_ep = te_cls.encode_prompt
        pipe = self.pipe

        def timed_ep(*args, **kw):
            if self._capture["on"]:
                self.captured_te_args = (args, kw)         # text-encoder trim gate
            b = event(torch)
            if self.te_offload:
                pipe.text_encoder.to(self.dev)
            out = orig_ep(*args, **kw)
            if self.te_offload:
                pipe.text_encoder.to("cpu")
                torch.cuda.empty_cache()
            self._stage.setdefault("text_encode", []).append((b, event(torch)))
            return out

        te_cls.encode_prompt = staticmethod(timed_ep)
        # split of the conditioner forward: Qwen3-VL vision tower vs language model (CUDA events; the rest of the text-encoder
        # step is the processor / tokenizer on the host)
        tm = getattr(self.pipe.text_encoder, "model", None)
        for sub, key in (("visual", "te_vision"), ("language_model", "te_language")):
            m = getattr(tm, sub, None)
            if m is not None:
                m.register_forward_pre_hook(lambda mod, a, kw, _k=key: self._te_open.__setitem__(_k, event(torch)),
                                            with_kwargs=True)
                m.register_forward_hook(lambda mod, a, kw, out, _k=key: self._stage.setdefault(_k, []).append(
                    (self._te_open.pop(_k), event(torch))), with_kwargs=True)

        if self.task == "ref2va":                  # video VAE encode of image / video refs + audio VAE encode of soundtracks
            orig_er = h3enc.MiniMaxH3Ref2VAReferenceEncoderStep.encode_references

            def timed_er(*args, **kw):
                if self._capture["on"]:
                    self.captured_er_args = args          # (components, prepared references, device): encode gate
                b = event(torch)
                out = orig_er(*args, **kw)
                self._stage.setdefault("reference_encode", []).append((b, event(torch)))
                return out

            h3enc.MiniMaxH3Ref2VAReferenceEncoderStep.encode_references = staticmethod(timed_er)
            # after the reference-encoder step: the conditioning every rank enters the layout with (capture only)
            rcls = h3enc.MiniMaxH3Ref2VAReferenceEncoderStep
            orig_call = rcls.__call__

            def call_then_text(inner, components, state, *a, **k):
                out = orig_call(inner, components, state, *a, **k)
                if self._capture["on"]:
                    self.captured_cond = {n: (None if state.get(n) is None else state.get(n).detach().clone())
                                          for n in ("prompt_embeds", "text_token_tags", "condition_latents",
                                                    "audio_condition_latents")}
                return out
            rcls.__call__ = call_then_text

        def wrap(mod, name):
            impl = mod.decode

            def wrapped(*args, **kw):
                self._depth[name] = self._depth.get(name, 0) + 1
                if self._depth[name] == 1 and self._capture["on"] and args:
                    self._captured[name] = args[0].detach().float().clone()
                    self.captured_raw[name] = args[0].detach().clone()
                b = event(torch) if self._depth[name] == 1 else None
                try:
                    out = impl(*args, **kw)
                finally:
                    self._depth[name] -= 1
                if b is not None:
                    self._stage.setdefault(name, []).append((b, event(torch)))
                return out
            mod.decode = wrapped

        wrap(self.pipe.vae, "video_vae_decode")
        wrap(self.pipe.audio_vae, "audio_vae_decode")

        if self.task == "ref2va":
            orig_enc = self.pipe.audio_vae.encode            # audio VAE encode of reference soundtracks / voice clips

            def timed_enc(*args, **kw):
                b = event(torch)
                out = orig_enc(*args, **kw)
                self._stage.setdefault("ref_audio_encode", []).append((b, event(torch)))
                return out
            self.pipe.audio_vae.encode = timed_enc
            self._install_prep_threads()
        self._install_block_timers()
        if self.ctl.sp is not None:
            self._install_comm_timers()

    def _install_prep_threads(self):
        """ref2va setup step: resize the reference video frames on `self.prep_threads` host threads. Same function body as
        diffusers packing_ref2va.prepare_reference_frames (one PIL LANCZOS resize per frame, np.stack in frame order), so the
        result is bitwise the stock one; PIL releases the GIL inside resize."""
        from concurrent.futures import ThreadPoolExecutor

        import numpy as np
        from diffusers.modular_pipelines.minimax_h3 import before_encoder as be
        from diffusers.modular_pipelines.minimax_h3 import packing_ref2va as pr
        from PIL import Image

        stock = be.prepare_reference_frames
        self._prep_stock = stock

        def threaded(frames, num_frames):
            n = self.prep_threads
            if n <= 0:
                return stock(frames, num_frames)
            if frames.ndim != 4 or frames.shape[3] != 3:
                return stock(frames, num_frames)          # stock raises its own error
            frames = frames[:num_frames]
            height, width = pr.resolve_canvas_size(frames.shape[2], frames.shape[1])
            if frames.shape[1:3] == (height, width):
                return frames

            def one(frame):
                return np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS))
            with ThreadPoolExecutor(max_workers=n) as ex:
                return np.stack(list(ex.map(one, frames)))
        be.prepare_reference_frames = threaded
        self._prep_threaded = threaded

    def _install_block_timers(self):
        """Stage audit: wrap the __call__ of every pipeline block class (top level + one level down, the denoise loop
        excluded) with a host wall clock between two device syncs. Active only while `self.audit` is set."""
        torch, seen = self.torch, set()

        def wrap_cls(blk, path):
            cls = type(blk)
            if cls in seen:
                return
            seen.add(cls)
            orig = cls.__call__

            def timed(inner, components, state, *a, _orig=orig, _path=path, **kw):
                if not self.audit:
                    return _orig(inner, components, state, *a, **kw)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                try:
                    return _orig(inner, components, state, *a, **kw)
                finally:
                    torch.cuda.synchronize()
                    self._blk.append((_path, (time.perf_counter() - t0) * 1e3))
            cls.__call__ = timed

        self._blk = []
        for name, blk in self.pipe._blocks.sub_blocks.items():
            wrap_cls(blk, name)
            if name != "denoise":
                for sub, b2 in getattr(blk, "sub_blocks", {}).items():
                    wrap_cls(b2, f"{name}.{sub}")

    def _install_comm_timers(self):
        """Stage audit: CUDA events around every SP collective (all-to-all, final all-gather; the compressed transport logs its
        own two all-to-alls) into ctl.comm_log; the time includes waiting for the slowest rank."""
        sp, ctl = self.ctl.sp, self.ctl
        for attr in ("a2a", "allgather"):
            f = getattr(sp, attr)

            def timed(*a, _f=f, _n=attr, **kw):
                return ctl._timed(_n, lambda: _f(*a, **kw))
            setattr(sp, attr, timed)

    # ---- one request -----------------------------------------------------------------------------------------------------------
    def generate(self, prompt, num_frames=124, seed=0, keep=False, capture_latents=False, references=None):
        """One pipeline call. Returns (record, media); media = (video [T, C, H, W] in [0, 1], audio, sample_rate, latents) if
        keep else None. e2e = host wall clock around the call with device syncs (and barriers when world > 1). ref2va needs
        `references` (list of diffusers MiniMaxH3Reference, in the order the model reads them)."""
        torch, ctl = self.torch, self.ctl
        if (self.task == "ref2va") != (references is not None):
            raise ValueError("references are required for ref2va and only valid there")
        self._captured.clear()
        if capture_latents:
            self.captured_raw = {}
        self._capture["on"] = capture_latents
        self._stage.clear()
        self._fwd.clear()
        self._blk = []
        self.ctl.comm_log = [] if self.audit else None
        ctl.counts = {}
        ctl.num_frames = num_frames
        req = dict(prompt=prompt.strip(), height=HEIGHT, width=WIDTH, num_frames=num_frames, num_inference_steps=SM.N_STEPS + 1,
                   generator=torch.Generator().manual_seed(int(seed)), output_type="pt")
        if references is not None:
            req["references"] = list(references)
        ctl.layout = None
        torch.cuda.synchronize()
        if self.world > 1:
            torch.distributed.barrier()
        torch.cuda.reset_peak_memory_stats(self.dev)
        h0 = time.perf_counter()
        with torch.no_grad():        # not inference_mode, as measured
            state = self.pipe(**req)
        torch.cuda.synchronize()
        if self.world > 1:
            torch.distributed.barrier()
        e2e_ms = (time.perf_counter() - h0) * 1e3

        videos, audio, sr = (_get(state, k) for k in ("videos", "audio", "sampling_rate"))
        st = {k: sum(b.elapsed_time(e) for b, e in v) for k, v in self._stage.items()}
        fw = [b.elapsed_time(e) for b, e in self._fwd]
        st["dit"] = sum(fw)
        peak = torch.tensor([torch.cuda.max_memory_allocated(self.dev), torch.cuda.max_memory_reserved(self.dev)],
                            dtype=torch.float64, device=self.dev)
        if self.world > 1:
            torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
        vid = videos[0]
        aud = None if audio is None else audio[0]
        ok = bool(torch.isfinite(vid.float()).all()) and (aud is None or bool(torch.isfinite(aud.float()).all()))
        rec = dict(preset=ctl.preset.name if ctl.preset else None, frames=num_frames, world=self.world, e2e_ms=e2e_ms,
                   dit_ms=st["dit"], stage_ms=st, forward_ms=fw, n_forwards=len(fw), counts=dict(ctl.counts), finite=ok,
                   video_shape=list(vid.shape), audio_shape=None if aud is None else list(aud.shape), sampling_rate=sr,
                   peak_alloc_gb=float(peak[0]) / 2**30, peak_reserved_gb=float(peak[1]) / 2**30, task=self.task,
                   layout=ctl.layout, kernel_options=dict(self.ckpt.int8.KERNEL_OPTIONS), int8_path=self.int8_path())
        if self.audit:
            rec["audit"] = self._gather_audit(e2e_ms, st)
        media = (vid, aud, sr, dict(self._captured)) if keep else None
        self._capture["on"] = False
        self._captured.clear()
        del state, videos, audio
        return rec, media


def _audit_rank(self, e2e_ms, st):
    comm, log = {}, self.ctl.comm_log or []
    for name, b, e in log:
        comm[name] = comm.get(name, 0.0) + b.elapsed_time(e)
    return dict(rank=self.rank, e2e_ms=e2e_ms, blocks=list(self._blk), stage_ms=dict(st), comm_ms=comm, n_comm=len(log),
                vae=None if self.pvae is None else self.pvae.describe(), prep_threads=self.prep_threads,
                transport=self.ctl.transport())


def _gather_audit(self, e2e_ms, st):
    """Per-rank stage records of the last request (all ranks; list ordered by rank)."""
    mine = self._audit_rank(e2e_ms, st)
    if self.world == 1:
        return [mine]
    out = [None] * self.world
    self.torch.distributed.all_gather_object(out, mine)
    return out


HyperFlowFlashRuntime._audit_rank = _audit_rank
HyperFlowFlashRuntime._gather_audit = _gather_audit


def per_rank_caches():
    """world > 1: give every rank its own compile-cache directories (seeded from the single-GPU caches when they exist), so
    concurrent Triton / CuTe DSL / inductor compiles never race on one directory. Call before importing torch."""
    import shutil

    rank, _, world = dist_env()
    if world <= 1:
        return
    for var in ("TRITON_CACHE_DIR", "CUTE_DSL_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH", "XDG_CACHE_HOME", "TMPDIR"):
        base = os.environ.get(var)
        if base:
            new = base.rstrip("/") + f"_rank{rank}"
            if os.path.isdir(base) and not os.path.isdir(new):
                shutil.copytree(base, new, dirs_exist_ok=True)
            os.makedirs(new, exist_ok=True)
            os.environ[var] = new
