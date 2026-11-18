"""Online RGB -> streaming TAE -> two full H3 evaluations -> causal RGB.

No reference MP4, full-video reference encode, or known final duration is used.
The initial picture and Qwen context are prepared once per session.
"""
import time
import torch
from dev.yanzuolu.common.seed import RandomState
from dev.yanzuolu.common.distributed.ops import get_device
from dev.yanzuolu.projects.minimax_h3.data.streaming import streaming_window_stop
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_prefix_dmd import MiniMaxH3VideoRefStreamingPrefixDMD
from .optimizations import STATE, install


class OnlineMeta(MiniMaxH3VideoRefStreamingPrefixDMD):
    def _configure_window_context(self, config):
        # Pictures come from the session, not a training dataset index.
        path = config.meta_model.get("qwen_processor_path")
        if not isinstance(path, str) or not path:
            raise ValueError("A Qwen processor deployment asset is required")
        self.qwen_processor_path = path
        self._qwen_processor_cache = None
        self.qwen_reference_video = bool(config.meta_model.get("qwen_reference_video", True))
        self.picture_mode = str(config.meta_model.get("picture_mode", "rows"))
        if self.qwen_reference_video or self.picture_mode not in {"rows", "keyframe"}:
            raise ValueError("Online runtime requires static picture+prompt conditioning")
        settings = dict(qwen_visual_context=True, qwen_processor_path=path,
                        fixed_window_rope=self.fixed_window_rope,
                        keep_sink_reference=self.keep_sink_reference,
                        qwen_reference_video=False)
        for key, value in settings.items():
            if key in config.data.args and config.data.args[key] != value:
                raise ValueError(f"data.args.{key} must match meta_model")
            config.data.args[key] = value

    def __init__(self, config):
        super().__init__(config)
        self.nfe = 0
        self.prefills = 0

    def _streaming_prefix_forward(self, *args, **kwargs):
        STATE.begin_forward()
        self.nfe += 1
        return super()._streaming_prefix_forward(*args, **kwargs)

    def _streaming_prefix_prefill(self, *args, **kwargs):
        STATE.begin_forward()
        self.prefills += 1
        return super()._streaming_prefix_prefill(*args, **kwargs)


class StreamingPipeline:
    def __init__(self, config, models, settings):
        install()
        self.meta = OnlineMeta(config)
        self.config = config
        self.models = models
        self.settings = settings
        self.device = get_device()
        self.rank = int(__import__("os").environ.get("RANK", "0"))
        self.states = None

    @torch.inference_mode()
    def start(self, *, prompt, picture, seed):
        STATE.reset()
        meta = self.meta
        if not meta.qwen_visual_context or meta.qwen_reference_video:
            raise ValueError("This deployment requires static prompt+picture Qwen conditioning")
        output, reference = self.settings["output"], self.settings["reference"]
        picture_rows = meta._validation_picture(
            self.models, {"picture": picture}, height=output["height"],
            width=output["width"], seed=seed)
        meta._stream_pictures = [picture_rows]
        meta._stream_qwen_contexts = meta._validation_qwen_contexts(
            self.models, [prompt], None, [picture_rows])
        self.states = meta.start_stream(
            prompt_embeds=[torch.empty((0, 0), device=self.device)],
            video_geometries=[(24, output["height"] // 16, output["width"] // 16)],
            reference_geometries=[(24, reference["height"] // 16, reference["width"] // 16)],
            audio_channels=[32], rngs=[RandomState(seed)],
            streaming_configs=[meta.streaming_config], prompts=[prompt])
        self.encoder = self.models["video_vae"].create_stream()
        self.decoder = None
        if self.rank == 0:
            policy = meta._streaming_policy(meta.streaming_config)
            self.decoder = meta._streaming_decoder(
                self.models, audio_lookahead_latents=policy["audio_lookahead_latents"],
                audio_right_lookahead_latents=policy.get("audio_right_lookahead_latents"))
        self.round = 0
        self.last_ready = None

    def next_counts(self):
        state = self.states[0]
        stop = streaming_window_stop(state.next_video, state.streaming_config,
                                     video_temporal_mapping=self.meta.video_temporal_mapping)
        boundary = self.meta.video_temporal_mapping.decode_timeline.boundary
        return stop - state.next_video, boundary(stop) - boundary(state.next_video)

    @torch.inference_mode()
    def render(self, rgb):
        """rgb is GPU-resident float32 CTHW [0,1], already reference-sized."""
        count, frames = self.next_counts()
        if rgb.shape != (3, frames, self.settings["reference"]["height"], self.settings["reference"]["width"]):
            raise ValueError(f"Reference chunk geometry mismatch: {tuple(rgb.shape)}")
        started = time.perf_counter()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        events[0].record()
        codec = self.models["video_vae"]
        value = rgb.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
        if not getattr(self.encoder, "requires_single_frame_encode_dispatch", False):
            raise RuntimeError("Encoder must enforce causal single-frame dispatch")
        pieces = []
        with torch.autocast("cuda", dtype=codec.param_dtype,
                            enabled=codec.param_dtype in (torch.float16, torch.bfloat16)):
            for i in range(frames):
                piece = self.encoder.encode(value[:, i:i + 1])
                if piece is not None:
                    if piece.shape[1] != 1:
                        raise RuntimeError("Unexpected encoder publication cadence")
                    pieces.append(piece)
        encoded = torch.cat(pieces, 1)[0].permute(1, 0, 2, 3).contiguous()
        encoded = ((encoded - codec.latents_mean.to(encoded).view(-1, 1, 1, 1))
                   / codec.latents_std.to(encoded).view(-1, 1, 1, 1)).contiguous()
        if encoded.shape[1] != count:
            raise RuntimeError(f"Expected {count} reference latents, got {encoded.shape[1]}")
        events[1].record()
        self.meta.nfe = self.meta.prefills = 0
        chunk = self.meta.step(self.models["backbone"], self.states, [encoded],
                               video_counts=[count], is_last=[False],
                               models=self.models, reference_pixels=[rgb])
        events[2].record()
        output = []
        if self.rank == 0:
            for i in range(chunk.video[0].shape[1]):
                publication = self.decoder.push(video=chunk.video[0][:, i:i + 1])
                if publication.video is not None:
                    output.append(publication.video)
        events[3].record()
        # Only chunk-end readiness; no synchronizations inserted between stages.
        events[3].synchronize()
        ready = time.perf_counter()
        if self.meta.nfe != 2:
            raise RuntimeError(f"Expected two full DiT NFEs, observed {self.meta.nfe}")
        result = None if not output else torch.cat(output, dim=2)
        metrics = {
            "round": self.round, "native_latents": count, "reference_frames": frames,
            "nfe": self.meta.nfe, "prefix_prefills": self.meta.prefills,
            "output_frames": 0 if result is None else int(result.shape[2]),
            "encode_ms": events[0].elapsed_time(events[1]),
            "render_ms": events[1].elapsed_time(events[2]),
            "decode_ms": events[2].elapsed_time(events[3]),
            "rgb_ready_gpu_ms": events[0].elapsed_time(events[3]),
            "rgb_ready_wall_ms": (ready - started) * 1000,
            "publication_interval_ms": None if self.last_ready is None else (ready - self.last_ready) * 1000,
        }
        self.last_ready = ready
        self.round += 1
        return result, metrics
