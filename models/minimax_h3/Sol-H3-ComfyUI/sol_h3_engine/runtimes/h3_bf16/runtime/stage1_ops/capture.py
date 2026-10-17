"""Task-local native H3 decode observer and existing AV media writer."""
from pathlib import Path
import time


def install_capture(worker):
    import torch
    from ltx_core.types import Audio
    from ..media_writer import encode_video
    from .native import atomic_json
    from fast_vae_adapter import Decoder, settings
    fast_config = settings()
    fast_decoder = None
    pipe = worker.pipeline
    state = {"target": None, "captures": 0, "rank0": True}
    worker._sol_h3_capture = state
    denoiser = pipe._stage_name_mapping["denoising_stage"]
    video_stage = pipe._stage_name_mapping["video_decoding_stage"]
    audio_stage = pipe._stage_name_mapping["audio_decoding_stage"]
    original_denoise, original_video, original_audio = denoiser.forward, video_stage.forward, audio_stage.forward

    def denoise(batch, args):
        torch.cuda.synchronize()
        started = time.monotonic_ns()
        result = original_denoise(batch, args)
        torch.cuda.synchronize()
        state["denoise_s"] = (time.monotonic_ns() - started) / 1e9
        return result

    def video_decode(batch, args):
        nonlocal fast_decoder
        if tuple(batch.raw_latent_shape) != (1, 24, 37, 48, 84):
            raise RuntimeError(f"Direct768 latent geometry mismatch: {batch.raw_latent_shape}")
        if args.video_decode_backend != "h3-vae" or args.output_type == "latent":
            raise RuntimeError("Direct output requires the original full H3 VAE")
        if not args.vae_cpu_offload or state["target"] is None:
            raise RuntimeError("Direct output requires VAE offload and an armed request")
        state["latent_shape"] = list(batch.raw_latent_shape)
        torch.cuda.synchronize()
        started = time.monotonic_ns()
        if fast_config:
            if fast_decoder is None:
                fast_decoder = Decoder(fast_config)
            result = fast_decoder.decode(batch, args)
        else:
            result = original_video(batch, args)
        torch.cuda.synchronize()
        state["video_decode_s"] = (time.monotonic_ns() - started) / 1e9
        if tuple(result.output.shape) != (1, 3, 124, 768, 1344) or result.output.device.type != "cpu":
            raise RuntimeError(f"Unexpected decoded output: {result.output.shape}")
        state["video_decode_calls"] += 1
        state["vae_placement_after_decode"] = (["cpu"] if fast_config else sorted({str(p.device) for p in video_stage.vae.parameters()}))
        if not fast_config and state["vae_placement_after_decode"] != ["cpu"]:
            raise RuntimeError("Native video VAE was not offloaded after decode")
        return result

    def audio_decode_and_write(batch, args):
        torch.cuda.synchronize()
        started = time.monotonic_ns()
        result = original_audio(batch, args)
        torch.cuda.synchronize()
        state["audio_decode_s"] = (time.monotonic_ns() - started) / 1e9
        pcm = result.extra["audio"]
        if pcm.dtype != torch.float32 or pcm.ndim != 2 or pcm.shape[1] != 2 or pcm.shape[0] < 161333:
            raise RuntimeError(f"Unexpected native PCM: {pcm.shape}")
        if result.extra["audio_sample_rate"] != 32000:
            raise RuntimeError("Native PCM sample rate drift")
        target = state["target"]
        output = Path(target["directory"]) / "native_768p.mp4"
        if output.exists() or batch.save_video or batch.return_frames:
            raise RuntimeError("Expected fresh output and task-owned mux")
        # Same released SDR/AAC encoder and16-frame writer chunks as the baselines.
        # Native H3 outputs CPU pixels; conversion/mux and any copy are timed.
        def chunks():
            for begin in range(0, 121, 16):
                chunk = result.output[0, :, begin:min(begin + 16, 121)].permute(1, 2, 3, 0)
                if not bool(torch.isfinite(chunk).all()):
                    raise RuntimeError("Nonfinite decoded H3 pixels")
                yield chunk
        started = time.monotonic_ns()
        audio = Audio(waveform=pcm[:161333].clamp(-1, 1).contiguous(), sampling_rate=32000)
        if not bool(torch.isfinite(audio.waveform).all()):
            raise RuntimeError("Nonfinite native PCM")
        encode_video(video=chunks(), fps=24, audio=audio, output_path=str(output),
                     video_chunks_number=8, crf=19, preset="veryfast", thread_count=8)
        finished = time.monotonic_ns()
        state["mux_s"] = (finished - started) / 1e9
        receipt = {
            **{k: target[k] for k in ("case_id", "prompt", "seed", "source_index")},
            "status": "PASS", "same_request": True, "external_anchor_used": False,
            "output": str(output), "output_bytes": output.stat().st_size,
            "final_mp4_complete_monotonic_ns": finished,
            "latent_shape": state["latent_shape"], "native_video_shape": [1, 3, 124, 768, 1344],
            "output_frames": 121, "fps": 24, "audio_samples": 161333, "audio_sample_rate": 32000,
            "latent_only_transfer": False, "video_decode_calls": state["video_decode_calls"],
            "h3_decoder_calls": state["video_decode_calls"], "decoder": "fast_noncausal" if fast_config else "original_H3_full_VAE",
            "vae_placement_after_decode": state["vae_placement_after_decode"],
            "phases_s": {key: state[key] for key in ("denoise_s", "video_decode_s", "audio_decode_s", "mux_s")},
            "video_decode_timing": ("noncausal student upload, eager engine build, decode, D2H, CPU RGB and offload"
                                    if fast_config else "native full stage including VAE upload, denormalize, decode, D2H and offload"),
            "media_writer": "existing pinned LTX media_io encoder only; no LTX model",
            "pixel_conversion_device": "cpu", "video_codec": "H264", "crf": 19, "preset": "veryfast",
        }
        atomic_json(output.parent / "capture.json", receipt)
        state["target"] = None
        state["captures"] += 1
        return result

    denoiser.forward, video_stage.forward, audio_stage.forward = denoise, video_decode, audio_decode_and_write
    return {"native_decode": True, "rank0": True}


def arm_capture(worker, target):
    state = worker._sol_h3_capture
    if state["target"] is not None:
        raise RuntimeError("Previous direct output incomplete")
    state["target"] = target
    state["video_decode_calls"] = 0
    return {"rank0": True, "captures": state["captures"]}
