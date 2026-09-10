"""Capture normalized H3 latents and native PCM from one prompt-only request."""
from pathlib import Path

def install_capture(worker):
    """Install when native lazy decode stages exist; never decode H3 video."""
    import hashlib
    import json
    import os
    from pathlib import Path
    import torch
    from fastvideo.distributed import get_world_group, model_parallel_is_initialized
    from fastvideo.pipelines.basic.minimax_h3.packing import h3_dit_patch_size, unpatchify_video_tokens
    from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import MINIMAX_H3_LAYOUT_KEY

    rank0 = not model_parallel_is_initialized() or get_world_group().is_first_rank
    state = {"target": None, "captures": 0, "rank0": bool(rank0), "video_decode_calls": 0}
    worker._sol_h3_capture = state
    video_stage = worker.pipeline._stage_name_mapping["video_decoding_stage"]
    audio_stage = worker.pipeline._stage_name_mapping["audio_decoding_stage"]
    original_audio = audio_stage.forward

    def digest(path):
        result = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                result.update(chunk)
        return result.hexdigest()

    def video_capture(batch, fastvideo_args):
        target = state["target"] if rank0 else None
        if target is not None:
            layout = batch.extra[MINIMAX_H3_LAYOUT_KEY]
            if layout.num_condition_video_rows or layout.num_condition_audio_rows:
                raise RuntimeError("T2VA capture forbids visual/audio condition rows")
            if fastvideo_args.output_type == "latent" or fastvideo_args.video_decode_backend != "h3-vae":
                raise RuntimeError("native audio decode configuration drift")
            if tuple(batch.raw_latent_shape) != (1, 24, 37, 24, 42):
                raise RuntimeError(f"unexpected H3 geometry: {batch.raw_latent_shape}")
            _, channels, frames, height, width = batch.raw_latent_shape
            native = unpatchify_video_tokens(
                batch.latents[layout.num_condition_video_rows:], frames, height, width,
                channels, h3_dit_patch_size(fastvideo_args)).detach().cpu().clone()
            if native.dtype != torch.float32 or not bool(torch.isfinite(native).all()):
                raise RuntimeError("expected finite native FP32 normalized H3 latent")
            state["native_normalized"] = native
        if not rank0 or target is None or batch.save_video or batch.return_frames:
            raise RuntimeError("latent capture requires rank zero and disabled video output")
        # output_type remains native: the audio stage must still decode PCM.
        batch.output = state["native_normalized"]
        return batch

    def audio_capture(batch, fastvideo_args):
        result = original_audio(batch, fastvideo_args)
        target = state["target"] if rank0 else None
        if target is not None:
            pcm = result.extra["audio"]
            if pcm.ndim != 2 or pcm.shape[1] != 2 or pcm.shape[0] < 161333:
                raise RuntimeError(f"unexpected native stereo PCM: {pcm.shape}")
            if pcm.dtype != torch.float32 or result.extra["audio_sample_rate"] != 32000:
                raise RuntimeError("native PCM dtype/sample rate drift")
            native = state.pop("native_normalized")
            video = native.to(torch.bfloat16).contiguous()
            audio = pcm[:161333].transpose(0, 1).unsqueeze(0).clamp(-1, 1).contiguous()
            if not bool(torch.isfinite(audio).all()):
                raise RuntimeError("nonfinite native PCM")
            directory = Path(target["directory"])
            payload = directory / "stage1_direct_tensors.pt"
            native_path = directory / "h3_normalized_native.pt"
            for path, value in ((payload, {"h3_normalized": video, "audio": audio}),
                                (native_path, {"h3_normalized": native})):
                with path.open("xb") as stream:
                    torch.save(value, stream)
            conversion_error = float((video.float() - native.float()).abs().max())
            receipt = {
                **{key: target[key] for key in ("case_id", "prompt", "seed", "source_index")},
                "request_id": target.get("request_id", target["case_id"]),
                "status": "PASS", "same_request": True, "external_anchor_used": False,
                "task": "t2va", "capture_rank": 0,
                "payload_path": str(payload), "payload_sha256": digest(payload),
                "generated_first_frame_path": None,
                "generated_first_frame_sha256": None,
                "native_normalized_path": str(native_path), "native_normalized_sha256": digest(native_path),
                "h3_normalized": {"shape": list(video.shape), "dtype": str(video.dtype),
                                  "source_dtype": str(native.dtype), "bf16_cast_max_abs_error": conversion_error,
                                  "normalization": "released_h3_per_channel_mean_std",
                                  "source": "original packed final latent unpatchified before VAE denormalize"},
                "audio": {"shape": list(audio.shape), "dtype": str(audio.dtype), "sample_rate": 32000,
                          "native_shape": list(pcm.shape), "temporal_trim": "first 161333 samples for 121-frame Stage2",
                          "processing": "transpose TC to BCT; clamp [-1,1], original native PCM unchanged"},
                "native_video_shape": None,
                "latent_only_transfer": True,
                "video_decode_calls": state["video_decode_calls"],
                "h3_decoder_calls": state["video_decode_calls"],
            }
            path = directory / "capture.json"
            temporary = path.with_suffix(".json.tmp")
            with temporary.open("x") as stream:
                json.dump(receipt, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
            os.replace(temporary, path)
            state["captures"] += 1
            state["target"] = None
        return result

    video_stage.forward = video_capture
    audio_stage.forward = audio_capture
    return {"rank0": bool(rank0), "video_stage": type(video_stage).__name__,
            "audio_stage": type(audio_stage).__name__}


def arm_capture(worker, target):
    state = worker._sol_h3_capture
    if state["target"] is not None or "native_normalized" in state:
        raise RuntimeError("previous request capture remains incomplete")
    # Only the output rank owns file writes; no nonzero-rank pending state.
    if state["rank0"]:
        state["target"] = target
        state["video_decode_calls"] = 0
    return {"rank0": state["rank0"], "captures": state["captures"]}
