"""HyperFlow-8 MiniMax-H3 on one 32 GB GPU, behind the same JSON-line protocol as worker.py.

Text conditioning reuses the verified NVFP4 Qwen session from the bundled runtime, in a
per-request subprocess (as the existing pipeline does), so the 62 GB BF16 Qwen3-VL is never
loaded. The DiT, VAEs and the official HyperFlow LoRA (PEFT, unmerged) stay resident in this
process: Diffusers >= 0.40 + hyperflow_h3, supplied by the profile's `hyperflow.deps` overlay.
The text-encoder blocks are removed exactly as in HyperFlow's official SplitPipeline; the DiT
streams one block group at a time; decode keeps the official fp16-autocast-over-fp32 recipe.

`t2va` and first-frame `fl2va` share one transformer partition, so one fl2va pipeline serves
both (a text request simply anchors no keyframes). First-frame conditioning runs locally: the
NVFP4 Qwen session presents the keyframe on the generation canvas exactly as the official
conditioner does (same stretch, same `"<Picture 1>: "` label and video-tagged vision block).
A request may instead carry a precomputed `conditioning` payload from the official conditioner.
"""
import json
import os
import sys
import time
from pathlib import Path

ENGINE = Path(__file__).resolve().parent
RUNTIME = ENGINE / "runtimes" / "h3_fp8"
FPS = 24
OUTPUT_FRAMES = 121
CANVAS = {"768p": (768, 1344), "480p": (480, 864)}
PHASES = {}  # seconds per staged encode/decode step of the current request (filled by the wrappers below)


class PinnedModule:
    """Keep a module's weights in pinned host memory and swap it on/off the card by repointing tensors.

    The VAEs never change at inference, so "offload" is only a pointer swap back to the pinned copy, and
    "onload" is one pinned→device copy per tensor (~0.2 s for the 9.8 GB video VAE). That leaves the card
    free for the per-request NVFP4 Qwen process between requests.
    """

    def __init__(self, module):
        import torch
        self.module = module
        self.host = {}
        for name, t in list(module.named_parameters()) + list(module.named_buffers()):
            self.host[name] = t.data.cpu().pin_memory() if not t.data.is_pinned() else t.data
            t.data = self.host[name]
        self.torch = torch
        self.on_gpu = False

    def _tensors(self):
        return list(self.module.named_parameters()) + list(self.module.named_buffers())

    def to_gpu(self):
        if not self.on_gpu:
            for name, t in self._tensors():
                t.data = self.host[name].to("cuda", non_blocking=True)
            self.torch.cuda.synchronize()
            self.on_gpu = True

    def to_host(self):
        if self.on_gpu:
            for name, t in self._tensors():
                t.data = self.host[name]
            self.on_gpu = False


def conditioned_blocks(workflow):
    """HyperFlow blocks without the steps our NVFP4 Qwen conditioning replaces.

    The text encoder always goes; `before_encode` goes for t2va/fl2va (the keyframe canvas is prepared before Qwen)
    but stays for ref2va, where it is MiniMaxH3Ref2VASetupStep: the request plan and normalized references.
    """
    from hyperflow_h3 import hyperflow_blocks
    blocks = hyperflow_blocks(workflow)
    for name in ("text_encoder",) if workflow == "ref2va" else ("before_encode", "text_encoder"):
        if name in blocks.sub_blocks:
            blocks.sub_blocks.pop(name)
    return blocks


def dump_qkv(path, block_index=0, step=0):
    """Diagnostics: save the post-RoPE q/k/v that one block's dense attention receives at one step."""
    import torch
    import diffusers.models.attention_dispatch as dispatch
    from hyperflow_h3.blocks import current_step
    original = dispatch.dispatch_attention_fn
    state = {"layer": -1, "done": False}

    def wrapped(query, key, value, *args, **kwargs):
        if not state["done"] and current_step() == step and query.shape[1] > 4096:
            state["layer"] += 1
            if state["layer"] == block_index:
                torch.save({"q": query.detach().cpu(), "k": key.detach().cpu(), "v": value.detach().cpu()}, path)
                state["done"] = True
        return original(query, key, value, *args, **kwargs)

    import hyperflow_h3.sol_attn as sol
    sol.dispatch_attention_fn = wrapped


def pin_unmatched_group(model):
    """Keep diffusers' block-level "unmatched group" (everything that is not a streamed block: token-refiner
    leftovers, embedders, norm_out, heads, top-level parameters) on the card for good.

    diffusers builds that group with stream=None and non_blocking=False, so every DiT forward moved it with a
    blocking .to() both ways (~2 GB of synchronous device<->host copies per step on H3). Onloading it once and
    turning its onload_/offload_ into no-ops keeps the prefetch chain of the streamed blocks intact.
    """
    from diffusers.hooks.group_offloading import _GROUP_OFFLOADING
    from diffusers.hooks.hooks import HookRegistry
    hook = HookRegistry.check_if_exists_or_initialize(model).get_hook(_GROUP_OFFLOADING)
    group = hook.group
    if group.stream is not None:
        raise RuntimeError("expected diffusers' unmatched group (stream=None) on the top-level DiT")
    group.onload_()
    group.onload_ = lambda: None
    group.offload_ = lambda: None
    return sum(p.numel() * p.element_size() for m in group.modules for p in m.parameters()) + \
        sum(p.numel() * p.element_size() for p in group.parameters)


class HyperFlow:
    def __init__(self, options, paths):
        overlay = options["hyperflow"]["deps"]
        sys.path.insert(0, str(Path(overlay).resolve(strict=True)))
        import torch
        from hyperflow_h3 import hyperflow_blocks, load_hyperflow_lora
        self.torch = torch
        self.paths = paths
        self.root = Path(options["work_dir"])
        self.started = time.monotonic()
        self.resident = bool(options["hyperflow"].get("resident"))
        model = paths["h3_model"]
        self.pipes = {}
        self.dit_pinned = {}
        transformer = vae = audio_vae = None
        workflows = list(options["hyperflow"].get("workflows", ["t2va"]))
        # t2va and fl2va read the same `transformer/` partition: build the fl2va pipeline once and
        # route text requests through it instead of holding a second 62 GB copy in host RAM.
        shared = "t2va" in workflows and "fl2va" in workflows and options["hyperflow"].get("share_fl2va_pipe", True)
        if shared:
            workflows.remove("t2va")
        self.options, self.model = options, model
        # swap_ref_dit: transformer_ref is built on the first reference request and swapped with `transformer`
        # (one FP8 DiT pinned in host RAM at a time: two exceed a 90 GB host).
        self.lazy_ref = "ref2va" in workflows and options["hyperflow"].get("swap_ref_dit", False)
        if self.lazy_ref:
            workflows.remove("ref2va")
        self.loaded = "main"
        for workflow in workflows:
            pipe = conditioned_blocks(workflow).init_pipeline(model, components_manager=None)
            name = "transformer_ref" if workflow == "ref2va" else "transformer"
            if options["hyperflow"].get("fp8_dir"):
                others = [n for n, spec in pipe._component_specs.items()
                          if spec.default_creation_method == "from_pretrained" and n not in ("transformer", "transformer_ref")]
                pipe.load_components(names=others, dtype=torch.bfloat16, pretrained_model_name_or_path=model,
                                     local_files_only=True)
                self.install_dit(pipe, workflow)
            else:
                pipe.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path=model,
                                     local_files_only=True)
                load_hyperflow_lora(pipe, options["hyperflow"]["weights"])
                self.configure_dit(pipe, workflow, getattr(pipe, name))
            # Share the decoders between workflows instead of holding two copies in host RAM.
            vae = vae or pipe.vae
            audio_vae = audio_vae or pipe.audio_vae
            pipe.audio_vae.set_attention_backend("native")
            # Resident: the whole pipeline lives on the card, so every step resolves its execution
            # device to cuda. Streamed: decoders wait on the host until the DiT has streamed out.
            for part in ("vae", "audio_vae"):
                getattr(pipe, part).to("cuda" if self.resident else "cpu")
            self.pipes[workflow] = pipe
        if shared:
            # Text requests get the t2va blocks over the fl2va pipeline's components (same DiT, decoders,
            # scheduler objects): no second copy of the weights, and the official t2va layout instead of an
            # empty keyframe list through the fl2va one.
            source = self.pipes["fl2va"]
            pipe = conditioned_blocks("t2va").init_pipeline(model, components_manager=None)
            pipe.update_components(**{name: getattr(source, name) for name in pipe._component_specs
                                      if getattr(source, name, None) is not None})
            from hyperflow_h3 import read_metadata
            from hyperflow_h3.blocks import configure_hyperflow_blocks
            configure_hyperflow_blocks(pipe.blocks, read_metadata(Path(options["hyperflow"]["weights"])))
            self.pipes["t2va"] = pipe
        # Decoders live in pinned host memory and visit the card only for encode/decode (resident: always there).
        self.pinned = {} if self.resident else {"vae": PinnedModule(vae), "audio_vae": PinnedModule(audio_vae)}
        self._stage_decoders()
        if os.environ.get("SOLH3_DUMP_QKV"):
            dump_qkv(os.environ["SOLH3_DUMP_QKV"])
        if options["hyperflow"].get("fast_vae"):
            self.load_fast_vae(options["hyperflow"]["fast_vae"])
        self.load_s = time.monotonic() - self.started

    def install_dit(self, pipe, workflow):
        """Build the FP8 HyperFlow DiT for `workflow` (transformer, or transformer_ref for ref2va) into `pipe`."""
        from hyperflow_h3.blocks import configure_hyperflow_blocks
        from hyperflow_fp8 import build_fp8_transformer
        flow, ref = self.options["hyperflow"], workflow == "ref2va"
        # LoRA merged + W8A8 block linears (convert_fp8.py / convert_ref.py) and the exact AdaLN table.
        dit, metadata = build_fp8_transformer(self.model, flow["ref_fp8_dir" if ref else "fp8_dir"], flow["weights"],
                                              flow.get("ref_adaln_table" if ref else "adaln_table"))
        pipe.update_components(**{"transformer_ref" if ref else "transformer": dit})
        configure_hyperflow_blocks(pipe.blocks, metadata)
        self.configure_dit(pipe, workflow, dit)
        return dit

    def configure_dit(self, pipe, workflow, dit):
        torch, flow = self.torch, self.options["hyperflow"]
        # Dense attention (HyperFlow's dense steps/blocks + the token refiner): SDPA, or SageAttention2++
        # (INT8 QK, FP8 PV with fp16 accumulation; diffusers backend `_sage_qk_int8_pv_fp8_cuda`).
        dit.set_attention_backend({"sage2pp": "_sage_qk_int8_pv_fp8_cuda"}.get(flow.get("dense_attn"), "native"))
        if flow.get("sol_attn"):
            # HyperFlow's own validated Sol-Attn recipe: dense for steps 1-2 and blocks 0-1, tau 1.0, diag.
            from hyperflow_h3 import enable_sol_attention
            enable_sol_attention(pipe)
        if flow.get("compile"):
            # Regional compile: one graph per block kind; attention stays an opaque eager op (hyperflow_compile).
            from hyperflow_compile import install as install_compile
            install_compile(dit)
        if self.resident:
            dit.to("cuda")  # cards with room for the whole BF16 DiT (e.g. 80 GB)
        elif flow.get("dit_onload"):
            # Whole DiT on the card for denoising only (FP8 + AdaLN table ≈ 21 GB fits a 32 GB card with the
            # 33k-token activations), parked in pinned host memory while Qwen encodes and the VAE decodes.
            dit.to("cpu")
            self.dit_pinned[workflow] = PinnedModule(dit)
        else:
            dit.enable_group_offload(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
                                     offload_type="block_level", num_blocks_per_group=1,
                                     use_stream=True, record_stream=True)
            if flow.get("pin_unmatched_group"):
                pin_unmatched_group(dit)

    def ensure_dit(self, task):
        """Swap the resident FP8 DiT when a request needs the other checkpoint partition (swap_ref_dit)."""
        if not self.lazy_ref:
            return
        want = "ref" if task == "ref2va" else "main"
        if want == self.loaded:
            return
        import gc
        torch = self.torch
        t0 = time.monotonic()
        main = [p for k, p in self.pipes.items() if k != "ref2va"]
        if want == "ref":
            for pipe in main:
                pipe.update_components(transformer=None)  # drop every reference to the main DiT first
            if "ref2va" not in self.pipes:
                pipe = conditioned_blocks("ref2va").init_pipeline(self.model, components_manager=None)
                source = main[0]
                pipe.update_components(**{n: getattr(source, n) for n in pipe._component_specs
                                          if n not in ("transformer", "transformer_ref") and getattr(source, n, None) is not None})
                pipe.audio_vae.set_attention_backend("native")
                self.pipes["ref2va"] = pipe
            gc.collect(); torch.cuda.empty_cache()
            self.install_dit(self.pipes["ref2va"], "ref2va")
        else:
            self.pipes["ref2va"].update_components(transformer_ref=None)
            gc.collect(); torch.cuda.empty_cache()
            dit = self.install_dit(main[0], "fl2va" if "fl2va" in self.pipes else "t2va")
            for pipe in main[1:]:
                pipe.update_components(transformer=dit)
        self.loaded = want
        PHASES["dit.swap"] = time.monotonic() - t0

    def _stage_decoders(self):
        torch = self.torch
        from diffusers.modular_pipelines.minimax_h3.decoders import (
            MiniMaxH3AudioDecodeStep, MiniMaxH3VideoDecodeStep)
        for cls in (MiniMaxH3VideoDecodeStep, MiniMaxH3AudioDecodeStep):
            if getattr(cls, "_solh3_staged", False):
                continue
            original = cls.__call__

            def staged(self, components, state, _original=original, _name=cls.__name__, _pinned=self.pinned,
                       _engine=self):
                # Denoising is over: give the card back from a resident DiT, then decode.
                for part in _engine.dit_pinned.values():
                    if part.on_gpu:
                        t0 = time.monotonic()
                        part.to_host()
                        torch.cuda.empty_cache()
                        PHASES["dit.to_host"] = PHASES.get("dit.to_host", 0.0) + time.monotonic() - t0
                dump = os.environ.get("SOLH3_DUMP_LATENTS")
                if dump and _name == "MiniMaxH3VideoDecodeStep":  # diagnostics: the exact latents the decoder sees
                    torch.save(self.get_block_state(state).latents.detach().cpu(), dump)
                if _name == "MiniMaxH3VideoDecodeStep" and _engine.fast_vae is not None:
                    return _engine.fast_video_decode(self, components, state)
                t0 = time.monotonic()
                for part in _pinned.values():
                    part.to_gpu()
                torch.cuda.synchronize()
                PHASES[_name + ".to_cuda"] = PHASES.get(_name + ".to_cuda", 0.0) + time.monotonic() - t0
                t0 = time.monotonic()
                with torch.autocast("cuda", dtype=torch.float16):
                    out = _original(self, components, state)
                torch.cuda.synchronize()
                PHASES[_name] = PHASES.get(_name, 0.0) + time.monotonic() - t0
                return out
            cls.__call__ = staged
            cls._solh3_staged = True
        # Keyframe / reference encoders run before the DiT streams in: borrow the card for the
        # video VAE (official fp32 encode, no autocast) and give it back before denoising.
        from diffusers.modular_pipelines.minimax_h3 import encoders
        for cls in (encoders.MiniMaxH3KeyframeVaeEncoderStep, encoders.MiniMaxH3Ref2VAReferenceEncoderStep):
            if getattr(cls, "_solh3_staged", False):
                continue
            original = cls.__call__

            def encode_staged(self, components, state, _original=original, _pinned=self.pinned, _name=cls.__name__):
                t0 = time.monotonic()
                if "vae" in _pinned:
                    _pinned["vae"].to_gpu()
                try:
                    return _original(self, components, state)
                finally:
                    torch.cuda.synchronize()
                    PHASES[_name] = PHASES.get(_name, 0.0) + time.monotonic() - t0
                    if "vae" in _pinned:
                        _pinned["vae"].to_host()
                        torch.cuda.empty_cache()
            cls.__call__ = encode_staged
            cls._solh3_staged = True

    fast_vae = None

    def load_fast_vae(self, config):
        """Distilled noncausal decoder (Efficient-Large-Model/Sol-H3-VAE): replaces only the video decode."""
        sys.path.insert(0, config["code"])
        import fa4_sm120
        from sol_h3_vae import SolH3VAEDecoder
        fa4_sm120.install()
        self.fast_vae = SolH3VAEDecoder(config["vae_config"], config["weights"], engine="fast", max_engines=1)
        self.fast_vae.model.to("cpu")
        self.fast_vae_pinned = PinnedModule(self.fast_vae.model)

    def fast_video_decode(self, step, components, state):
        """Same contract as MiniMaxH3VideoDecodeStep: normalized latents in, pixels in [0, 1] out (the student
        denormalizes internally, so the latents are handed over as-is), then the pipeline's own postprocess."""
        torch = self.torch
        block_state = step.get_block_state(state)
        t0 = time.monotonic()
        self.fast_vae_pinned.to_gpu()
        try:
            latents = block_state.latents  # (1, 24, T, H, W), normalized
            # Eager engine per request: a captured CUDA graph would keep the weight addresses of this visit, and
            # the weights go back to pinned host memory afterwards (the decoder alone peaks near 30 GB at 768p).
            from sol_h3_vae.fast import NonCausalFast3
            _, _, frames, height, width = latents.shape
            engine = NonCausalFast3(self.fast_vae.model, frames, height, width, cuda_graph=False)
            pixels = engine.decode(latents.transpose(1, 2).float().contiguous())
            del engine
            device = pixels.device
            pixel_mean = torch.tensor(components.pixel_mean, device=device).view(1, -1, 1, 1, 1)
            pixel_std = torch.tensor(components.pixel_std, device=device).view(1, -1, 1, 1, 1)
            video = (pixels.float() * pixel_std + pixel_mean).clamp(0, 1)
            block_state.videos = components.video_processor.postprocess_video(video, output_type=block_state.output_type)
            torch.cuda.synchronize()
        finally:
            self.fast_vae_pinned.to_host()
            torch.cuda.empty_cache()
        PHASES["FastVideoDecode"] = time.monotonic() - t0
        step.set_block_state(state, block_state)
        return components, state

    def generate(self, case, payload, out_dir, progress):
        torch = self.torch
        task = case.get("task", "t2va")
        swap_started = time.monotonic()
        self.ensure_dit(task)
        swap_s = time.monotonic() - swap_started
        pipe = self.pipes[task]
        dit = pipe.transformer_ref if task == "ref2va" else pipe.transformer
        steps = [0]

        def count_step(*_):
            # A forward pre-hook must return None: any other value replaces the module's inputs.
            steps[0] += 1
            progress("denoising", steps[0], 8)
        hook = dit.register_forward_pre_hook(count_step)
        # Everything the official conditioner left in the state (keyframes on the canvas, anchors,
        # resolved size, tags) goes to the remaining blocks unchanged, as in SplitPipeline.
        kwargs = {k: v for k, v in payload.items() if k not in ("task", "prompt", "input_spec",
                                                                "prepared_media", "external_anchor_used",
                                                                "input_conditioned")}
        kwargs["prompt_embeds"] = payload["prompt_embeds"].to("cuda", torch.bfloat16)
        kwargs["text_token_tags"] = payload["text_token_tags"].to(torch.long)
        kwargs.setdefault("height", 768)
        kwargs.setdefault("width", 1344)
        if task == "ref2va":
            from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3ImageReference
            kwargs["references"] = [MiniMaxH3ImageReference.from_file(path) for path in kwargs.pop("reference_images")]
        kwargs.update(num_frames=124, output_type="pt", generator=torch.Generator().manual_seed(case["seed"]))
        PHASES.clear()
        if swap_s > 0.5:
            PHASES["dit.swap"] = swap_s
        parked = self.dit_pinned.get(task) or (self.dit_pinned.get("fl2va") if task == "t2va" else None)
        if parked is not None:
            t0 = time.monotonic()
            parked.to_gpu()  # pinned → card, one copy per request (decode parks it again)
            PHASES["dit.to_gpu"] = time.monotonic() - t0
        try:
            started = time.monotonic()
            result = pipe(**kwargs, output=["videos", "audio", "sampling_rate"])
            torch.cuda.synchronize()
        finally:
            hook.remove()
            t0 = time.monotonic()
            for part in self.pinned.values():
                part.to_host()
            torch.cuda.empty_cache()
            PHASES["decoders.to_cpu"] = time.monotonic() - t0
        if steps[0] != 8:
            raise RuntimeError("HyperFlow must run exactly 8 DiT forwards, got %d" % steps[0])
        progress("saving", 8, 8)
        video, audio = result["videos"][0], result["audio"][0]
        if not (bool(torch.isfinite(video).all()) and bool(torch.isfinite(audio).all())):
            raise RuntimeError("non-finite HyperFlow output")
        output = Path(out_dir) / "hyperflow.mp4"
        t0 = time.monotonic()
        write_mp4(output, video[:OUTPUT_FRAMES], audio, result["sampling_rate"])
        PHASES["write_mp4"] = time.monotonic() - t0
        return {"output": str(output), "generate_s": time.monotonic() - started, "forwards": steps[0],
                "phases_s": {k: round(v, 2) for k, v in PHASES.items()}}


def write_mp4(path, video, audio, rate):
    import av
    with av.open(str(path), "w") as container:
        vs = container.add_stream("libx264", rate=FPS)
        vs.height, vs.width = video.shape[-2:]
        vs.pix_fmt = "yuv420p"
        vs.options = {"crf": "18", "preset": "veryfast"}  # medium cost ~10 s per 5 s clip at no visible gain
        au = container.add_stream("aac", rate=rate)
        au.layout = "stereo"
        for index, frame in enumerate(video):
            rgb = (frame.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
            packet_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            packet_frame.pts = index
            for packet in vs.encode(packet_frame):
                container.mux(packet)
        for packet in vs.encode():
            container.mux(packet)
        samples = av.AudioFrame.from_ndarray(audio.float().cpu().numpy(), format="fltp", layout="stereo")
        samples.sample_rate, samples.pts = rate, 0
        for packet in au.encode(samples):
            container.mux(packet)
        for packet in au.encode():
            container.mux(packet)


def local_conditioning(case, request, request_dir, paths, config, Worker, qwen=None):
    """NVFP4 Qwen conditioning for a text, first-frame or reference request, plus the state the rest needs."""
    import torch
    sys.path.insert(0, str(RUNTIME))
    from runtime.qwen_ops.media import load_rgb, prepare_keyframe_image
    mode, images = request.get("mode", "text"), list(request.get("images") or [])
    # 16:9 canvases, both axes multiples of 32 (the VAE's 16 x the 2x2 patch).
    height, width = CANVAS[request.get("resolution", "768p")]
    qcase = dict(case)
    keyframes = []
    if mode == "first_frame":
        if len(images) != 1:
            raise ValueError("First frame mode needs exactly one image")
        # Same canvas and arithmetic as the official ResizeStep: the keyframe is stretched onto
        # the canvas, and those exact pixels feed both Qwen's vision block and the VAE encoder.
        qcase.update(task="fl2va", first_frame=images[0], keyframe_canvas=(height, width))
        keyframes = [prepare_keyframe_image(load_rgb(images[0]), height, width, stretch=True)]
    elif mode == "reference":
        if not 1 <= len(images) <= 9:
            raise ValueError("Reference mode needs 1 to 9 images")
        # Qwen sees each reference at the canvas's pixel budget (Sol-H3 Spark ref2va "match"), not at a 2048 short
        # edge: two 2048-edge images overflow the card beside the 15.7 GB NVFP4 encoder.
        qcase.update(task="ref2va", references=[{"type": "image", "path": path} for path in images],
                     reference_pixel_budget=height * width)
    elif mode != "text":
        raise ValueError("Unknown mode: %s" % mode)
    if qwen is None:  # one process per request (original lifetime)
        qwen = Worker("qwen", request_dir / "qwen-worker", paths, config)
        try:
            receipt = qwen.call(case=qcase, output_root=str(request_dir / "qwen"))
        finally:
            qwen.close()
    else:  # resident process: weights parked in pinned host RAM between requests
        torch.cuda.empty_cache()  # the DiT's cached blocks must not crowd the encoder
        qwen.call(op="onload")
        try:
            receipt = qwen.call(case=qcase, output_root=str(request_dir / "qwen"))
        finally:
            qwen.call(op="offload")
    payload = torch.load(Path(receipt["conditioning_path"]).parent / "conditioning.pt",
                         map_location="cpu", weights_only=True)
    out = {"task": qcase["task"], "prompt": case["prompt"],
           "prompt_embeds": payload["prompt_embeds"], "text_token_tags": payload["text_token_tags"],
           "height": height, "width": width}
    if keyframes:
        out.update(keyframes=keyframes, keyframe_anchors=("first",))
    if mode == "reference":
        out["reference_images"] = images
    return out, receipt["request_wall_s"]


def serve(options, write_json):
    """JSON-line loop; mirrors worker.py's request/response contract."""
    import signal
    sys.path.insert(0, str(RUNTIME))
    from runtime.config import load_paths, load_recipe
    from runtime.pipeline import Worker
    paths = load_paths(options["paths"])
    config = load_recipe()
    config["execution"] = {"offload": "cpu", "gpu_count": 1, "qwen_lifetime": "one_request",
                           "overlap_stages": False}
    root = Path(options["work_dir"])
    engine = None
    qwen = None  # resident Qwen worker (hyperflow.qwen_resident): loaded once, parked in pinned host RAM when idle
    stopped = False

    def stop(_sig, _frame):
        nonlocal stopped
        if not stopped:
            stopped = True
            raise KeyboardInterrupt("Closing this ComfyUI worker")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request["op"] == "close":
                break
            if request["op"] != "generate":
                raise ValueError("Unsupported operation")
            progress_path = request.get("progress")

            def progress(stage, step=None, steps=None):
                if progress_path:
                    write_json(progress_path, {"stage": stage, "step": step, "steps": steps,
                                               "time": time.time()})
            try:
                started = time.monotonic_ns()
                case = {"case_id": request["case_id"], "prompt": request["prompt"],
                        "seed": request["seed"], "task": "t2va"}
                request_dir = root / case["case_id"]
                request_dir.mkdir(parents=True, exist_ok=False)
                cold = engine is None and request.get("op_detail") != "conditioning_only"
                if cold:
                    progress("loading")
                    engine = HyperFlow(options, paths)
                    if options["hyperflow"].get("qwen_resident"):
                        load_started = time.monotonic()
                        qwen = Worker("qwen", root / "qwen-resident", paths, config)
                        qwen.call(op="offload")
                        engine.load_s += time.monotonic() - load_started
                if request.get("conditioning"):
                    import torch
                    payload = torch.load(request["conditioning"], map_location="cpu", weights_only=False)
                    case["task"] = payload.get("task", "t2va")
                    qwen_s = qwen_call_s = 0.0
                else:
                    progress("conditioning")
                    qwen_started = time.monotonic()
                    payload, qwen_call_s = local_conditioning(case, request, request_dir, paths, config, Worker, qwen)
                    # Full Qwen wall time of this request (process start + load, or onload/offload), not only encode.
                    qwen_s = time.monotonic() - qwen_started
                    case["task"] = payload["task"]
                if request.get("op_detail") == "conditioning_only":
                    import torch
                    torch.save(payload, request_dir / "payload.pt")
                    write_json(request["response"], {"status": "PASS", "payload": str(request_dir / "payload.pt"),
                                                     "qwen_s": qwen_s, "task": case["task"]})
                    continue
                row = engine.generate(case, payload, request_dir, progress)
                e2e = (time.monotonic_ns() - started) / 1e9
                progress("done", 8, 8)
                write_json(request["response"], {
                    "status": "PASS", "case_id": case["case_id"], "seed": case["seed"],
                    "task": case["task"], "output": row["output"], "e2e_s": e2e,
                    "qwen_s": qwen_s, "qwen_encode_s": qwen_call_s if not request.get("conditioning") else 0.0,
                    "h3_phases_s": {"generate_s": row["generate_s"], **row.get("phases_s", {})},
                    "forwards": row["forwards"], "profile": options["name"], "precision": "hyperflow",
                    "decoder": "original", "single_stage": True, "cold_start": cold,
                    "startup_and_warmup_s": engine.load_s if cold else 0, "prompt": case["prompt"],
                    "timing": "request through completed MP4; model load reported separately"})
            except Exception as error:
                write_json(request["response"], {"status": "FAIL",
                                                 "error": "%s: %s" % (type(error).__name__, error)})
                raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if qwen is not None:
            qwen.close()
