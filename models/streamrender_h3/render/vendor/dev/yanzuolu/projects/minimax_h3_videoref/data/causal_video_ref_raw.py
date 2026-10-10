"""Stateful raw corpus for causal video-reference T2AV training."""

from __future__ import annotations

import json
import math
import pickle
import random
import subprocess
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import get_worker_info

from dev.yanzuolu.common.data import (
    WorkerResumeContext,
    WorkerStateEnvelope,
)
from dev.yanzuolu.common.seed import yield_seed
from dev.yanzuolu.projects.minimax_h3_videoref.data.causal_video_ref_latent import (
    _CausalVideoRefDataset,
    build_causal_video_ref_layout,
)
from dev.yanzuolu.projects.minimax_h3.modeling.constants import MINIMAX_H3_SUPPORTED_FPS

_AUDIO_SAMPLE_RATE = 32000
_AUDIO_HOP_LENGTH = 800
_SPATIAL_VAE_STRIDE = 16
_SPATIAL_PATCH = 2
_FFPROBE_TIMEOUT_SECONDS = 120
_FFMPEG_TIMEOUT_SECONDS = 600
_SOURCE_FIELDS = (
    "index_path",
    "data_root",
    "ratio",
    "ref_field",
    "video_field",
    "prompt_field",
)


def _require_source_path(value: Any, field: str, source_index: int) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"sources[{source_index}].{field} must be a non-empty string")
    return value


def _resolve_data_path(data_root: Path, raw_path: Any, field: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"JSONL field {field!r} must be a non-empty path string")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = data_root / path
    if path.is_symlink():
        raise ValueError(f"JSONL field {field!r} must not name a symlink: {path}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(
            f"JSONL field {field!r} resolves outside data_root: {resolved}"
        ) from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(
            f"JSONL field {field!r} must resolve to a regular non-symlink file: "
            f"{resolved}"
        )
    return resolved


def _read_prompt_file(path: Path) -> str:
    payload = path.read_bytes()
    try:
        text = payload.decode("utf8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"prompt file must be UTF-8: {path}") from exc
    suffix = path.suffix.lower()
    if suffix == ".txt":
        prompt = text.strip()
        if not prompt:
            raise ValueError(f"prompt text must be non-empty: {path}")
    elif suffix == ".json":
        try:
            document = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"prompt JSON is invalid: {path}") from exc
        if not isinstance(document, dict):
            raise ValueError(f"prompt JSON must be an object: {path}")
        raw_prompt = document.get("h3_prompt")
        if not isinstance(raw_prompt, str) or not raw_prompt.strip():
            raise ValueError(f"prompt JSON h3_prompt must be non-empty: {path}")
        prompt = raw_prompt.strip()
    else:
        raise ValueError(f"unsupported prompt file suffix {path.suffix!r}: {path}")
    return prompt


def _run_json_probe(command: list[str], path: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_FFPROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"FFprobe timed out after {_FFPROBE_TIMEOUT_SECONDS}s while probing {path}"
        ) from exc
    if result.returncode != 0:
        stderr = result.stderr.decode("utf8", "replace")
        raise RuntimeError(f"FFprobe failed for {path}:\n{stderr}")
    try:
        document = json.loads(result.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"FFprobe returned invalid JSON for {path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"FFprobe result must be an object for {path}")
    return document


def _positive_decimal(value: Any) -> Decimal | None:
    if value is None or value == "N/A":
        return None
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        return None
    if not result.is_finite() or result <= 0:
        return None
    return result


def _positive_fraction(value: Any) -> Fraction | None:
    if value is None or value == "N/A":
        return None
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        return None
    if result <= 0:
        return None
    return result


def _stream_duration(stream: Mapping[str, Any]) -> Fraction | None:
    durations: list[Fraction] = []
    duration = _positive_fraction(stream.get("duration"))
    if duration is not None:
        durations.append(duration)
    duration_ts = _positive_fraction(stream.get("duration_ts"))
    time_base = _positive_fraction(stream.get("time_base"))
    if duration_ts is not None and time_base is not None:
        durations.append(duration_ts * time_base)
    return max(durations) if durations else None


def _presentation_duration(
    records: Any,
    *,
    timestamp_fields: tuple[str, ...],
    duration_fields: tuple[str, ...],
) -> Decimal | None:
    if not isinstance(records, list):
        return None
    presentations: list[tuple[Decimal, Decimal | None]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        timestamp = None
        for field in timestamp_fields:
            timestamp = _positive_decimal(record.get(field))
            if timestamp is not None:
                break
            raw_timestamp = record.get(field)
            if raw_timestamp is not None and raw_timestamp != "N/A":
                try:
                    candidate = Decimal(str(raw_timestamp))
                except InvalidOperation:
                    continue
                if candidate.is_finite():
                    timestamp = candidate
                    break
        if timestamp is None:
            continue
        interval = None
        for field in duration_fields:
            interval = _positive_decimal(record.get(field))
            if interval is not None:
                break
        presentations.append((timestamp, interval))
    if not presentations:
        return None
    presentations.sort(key=lambda item: item[0])
    first_timestamp = presentations[0][0]
    last_timestamp, last_interval = presentations[-1]
    if last_interval is None:
        distinct_timestamps = sorted({timestamp for timestamp, _ in presentations})
        if len(distinct_timestamps) < 2:
            return None
        last_interval = distinct_timestamps[-1] - distinct_timestamps[-2]
        if last_interval <= 0:
            return None
    duration = last_timestamp - first_timestamp + last_interval
    return duration if duration > 0 else None


def _scan_video_duration(path: Path) -> Decimal:
    # This VFR fallback scans timestamps only to determine video duration.
    # Media windows are still decoded by the bounded FFmpeg commands below.
    frame_probe = _run_json_probe(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time,pts_time,pkt_duration_time,duration_time",
            "-of",
            "json",
            str(path),
        ],
        path,
    )
    duration = _presentation_duration(
        frame_probe.get("frames"),
        timestamp_fields=("best_effort_timestamp_time", "pts_time"),
        duration_fields=("pkt_duration_time", "duration_time"),
    )
    if duration is not None:
        return duration
    packet_probe = _run_json_probe(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_packets",
            "-show_entries",
            "packet=pts_time,duration_time",
            "-of",
            "json",
            str(path),
        ],
        path,
    )
    duration = _presentation_duration(
        packet_probe.get("packets"),
        timestamp_fields=("pts_time",),
        duration_fields=("duration_time",),
    )
    if duration is None:
        raise ValueError(f"FFprobe could not determine video presentation duration: {path}")
    return duration


def _probe_video_duration(path: Path) -> Fraction:
    document = _run_json_probe(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=time_base,duration,duration_ts,nb_frames,r_frame_rate,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        path,
    )
    streams = document.get("streams")
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], dict):
        raise ValueError(f"FFprobe must report exactly one selected video stream: {path}")
    stream = streams[0]
    r_frame_rate = _positive_fraction(stream.get("r_frame_rate"))
    avg_frame_rate = _positive_fraction(stream.get("avg_frame_rate"))
    try:
        nb_frames = int(stream.get("nb_frames"))
    except (TypeError, ValueError):
        nb_frames = 0
    if (
        nb_frames > 0
        and r_frame_rate is not None
        and avg_frame_rate is not None
        and r_frame_rate == avg_frame_rate
    ):
        frame_duration = Fraction(nb_frames) / r_frame_rate
        duration = _stream_duration(stream)
        return max(frame_duration, duration) if duration is not None else frame_duration
    return Fraction(_scan_video_duration(path))


def _run_ffmpeg(command: list[str], description: str) -> bytes:
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_FFMPEG_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"FFmpeg timed out after {_FFMPEG_TIMEOUT_SECONDS}s while decoding "
            f"{description}"
        ) from exc
    if result.returncode != 0:
        stderr = result.stderr.decode("utf8", "replace")
        raise RuntimeError(f"FFmpeg failed to decode {description}:\n{stderr}")
    return result.stdout


def _decode_video_bytes(
    payload: bytes,
    *,
    path: Path,
    num_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    expected_bytes = num_frames * height * width * 3
    if len(payload) != expected_bytes:
        frame_bytes = height * width * 3
        decoded_frames, remainder = divmod(len(payload), frame_bytes)
        detail = f"{decoded_frames} frames"
        if remainder:
            detail += f" plus {remainder} bytes"
        raise RuntimeError(
            f"FFmpeg decoded {detail} from {path}, expected exactly {num_frames} frames"
        )
    buffer = bytearray(payload)
    del payload
    video_view = torch.frombuffer(buffer, dtype=torch.uint8)
    video = (
        video_view.reshape(num_frames, height, width, 3)
        .permute(3, 0, 1, 2)
        .contiguous()
    )
    del video_view
    del buffer
    return video.to(torch.float32).div_(255)


def _decode_video_on_frame_grid(
    path: Path,
    *,
    start_frame: int,
    num_frames: int,
    fps: int,
    height: int,
    width: int,
    spatial_filter: str,
    description: str,
) -> torch.Tensor:
    """Resample from the video origin, then crop one integer frame-grid window."""
    assert isinstance(start_frame, int) and not isinstance(start_frame, bool) and start_frame >= 0
    filters = (
        f"fps={fps},trim=start_frame={start_frame}:end_frame={start_frame + num_frames},"
        f"setpts=PTS-STARTPTS,{spatial_filter},setsar=1"
    )
    payload = _run_ffmpeg(
        [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "1",
            "-filter_threads",
            "1",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-vf",
            filters,
            "-frames:v",
            str(num_frames),
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        description,
    )
    return _decode_video_bytes(
        payload,
        path=path,
        num_frames=num_frames,
        height=height,
        width=width,
    )


def decode_reference_video(
    path: Path,
    *,
    start_frame: int,
    num_frames: int,
    fps: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Decode a reference frame-grid window as float32 ``CTHW`` pixels in ``[0, 1]``."""
    return _decode_video_on_frame_grid(
        path,
        start_frame=start_frame,
        num_frames=num_frames,
        fps=fps,
        height=height,
        width=width,
        spatial_filter=f"scale={width}:{height}:flags=lanczos",
        description=f"reference video from {path}",
    )


class CausalVideoRefRawT2AVDataset(_CausalVideoRefDataset):
    """Raw target-video/audio and depth-reference stream with resumable packing."""

    _STATE_SCHEMA = "minimax_h3_videoref_causal_video_ref_raw_df_worker"
    _STATE_VERSION = 4
    forcing = "diffusion"

    def __init__(
        self,
        seed: int,
        resume_context: WorkerResumeContext,
        *,
        sources: Sequence[Mapping[str, Any]],
        num_frames: int,
        height: int,
        width: int,
        fps: int,
        reference_height: int,
        reference_width: int,
        **kwargs: Any,
    ) -> None:
        if fps != MINIMAX_H3_SUPPORTED_FPS:
            raise ValueError(
                f"fps must equal MINIMAX_H3_SUPPORTED_FPS="
                f"{MINIMAX_H3_SUPPORTED_FPS}, got {fps}"
            )
        if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
            raise ValueError("sources must be a non-empty sequence")
        source_list = list(sources)
        if not source_list:
            raise ValueError("sources must be a non-empty sequence")

        self.entries: list[dict[str, Path]] = []
        self._prompt_cache: dict[Path, str] = {}
        expected_source_fields = set(_SOURCE_FIELDS)
        for source_index, source in enumerate(source_list):
            if not isinstance(source, Mapping) or set(source) != expected_source_fields:
                raise ValueError(
                    f"sources[{source_index}] fields must be exactly "
                    f"{list(_SOURCE_FIELDS)}"
                )
            raw_index_path = _require_source_path(
                source["index_path"], "index_path", source_index
            )
            raw_data_root = _require_source_path(
                source["data_root"], "data_root", source_index
            )
            index_path = Path(raw_index_path).expanduser().resolve(strict=True)
            if not index_path.is_file():
                raise ValueError(f"sources[{source_index}].index_path must be a file")
            data_root = Path(raw_data_root).expanduser().resolve(strict=True)
            if not data_root.is_dir():
                raise ValueError(f"sources[{source_index}].data_root must be a directory")
            try:
                ratio = Decimal(str(source["ratio"]))
            except InvalidOperation as exc:
                raise ValueError(
                    f"sources[{source_index}].ratio must satisfy 0 < ratio <= 1"
                ) from exc
            if not ratio.is_finite() or ratio <= 0 or ratio > 1:
                raise ValueError(
                    f"sources[{source_index}].ratio must satisfy 0 < ratio <= 1"
                )
            ref_field = _require_source_path(
                source["ref_field"], "ref_field", source_index
            )
            video_field = _require_source_path(
                source["video_field"], "video_field", source_index
            )
            prompt_field = _require_source_path(
                source["prompt_field"], "prompt_field", source_index
            )

            index_payload = index_path.read_bytes()
            nonempty_lines = [line for line in index_payload.splitlines() if line.strip()]
            retained_count = int(
                (Decimal(len(nonempty_lines)) * ratio).to_integral_value(
                    rounding=ROUND_FLOOR
                )
            )
            for retained_index, line in enumerate(
                nonempty_lines[:retained_count], start=1
            ):
                try:
                    record = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"{index_path}: retained non-empty line {retained_index}: "
                        "invalid JSON"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"{index_path}: retained non-empty line {retained_index}: "
                        "entry must be an object"
                    )
                reference_path = _resolve_data_path(
                    data_root, record.get(ref_field), ref_field
                )
                video_path = _resolve_data_path(
                    data_root, record.get(video_field), video_field
                )
                prompt_path = _resolve_data_path(
                    data_root, record.get(prompt_field), prompt_field
                )
                if prompt_path not in self._prompt_cache:
                    self._prompt_cache[prompt_path] = _read_prompt_file(prompt_path)
                self.entries.append(
                    {
                        "reference_path": reference_path,
                        "video_path": video_path,
                        "prompt_path": prompt_path,
                    }
                )
        if not self.entries:
            raise ValueError("sources retain no entries")

        super().__init__(
            seed,
            resume_context,
            prompts=("raw video reference",),
            default_prompt="raw video reference",
            num_frames=num_frames,
            height=height,
            width=width,
            **kwargs,
        )
        self.fps = int(fps)
        self.reference_height = int(reference_height)
        self.reference_width = int(reference_width)
        if (
            self.height <= 0
            or self.width <= 0
            or self.height % (_SPATIAL_VAE_STRIDE * _SPATIAL_PATCH)
            or self.width % (_SPATIAL_VAE_STRIDE * _SPATIAL_PATCH)
        ):
            raise ValueError("height and width must be positive and divisible by 32")
        if (
            self.reference_height <= 0
            or self.reference_width <= 0
            or self.reference_height % (_SPATIAL_VAE_STRIDE * _SPATIAL_PATCH)
            or self.reference_width % (_SPATIAL_VAE_STRIDE * _SPATIAL_PATCH)
        ):
            raise ValueError(
                "reference_height and reference_width must be positive and divisible by 32"
            )
        self.reference_latent_shape = (
            self.video_latent_channels,
            self.latent_t,
            self.reference_height // _SPATIAL_VAE_STRIDE,
            self.reference_width // _SPATIAL_VAE_STRIDE,
        )
        self._video_duration_cache: dict[Path, Fraction] = {}

    def _encode_worker_state(
        self,
        logical_worker_id: int,
        offset: int,
        avg_seqlen: float,
        cnt: int,
    ) -> bytes:
        return pickle.dumps(
            {
                "schema": self._STATE_SCHEMA,
                "version": self._STATE_VERSION,
                "logical_worker_id": logical_worker_id,
                "offset": offset,
                "avg_seqlen": avg_seqlen,
                "cnt": cnt,
            }
        )

    def _pack_video_ref_sample(self, *, prompt: str) -> dict[str, Any]:
        text_input_ids, text_len = self.tokenizer.encode(prompt)
        return {
            "prompts": prompt,
            "text_input_ids": text_input_ids,
            "text_lens": text_len,
            **build_causal_video_ref_layout(
                text_len=text_len,
                target_video_shape=self.latent_shape,
                target_audio_shape=self.audio_shape,
                reference_video_shape=self.reference_latent_shape,
                forcing=self.forcing,
                chunk_size=self.chunk_size,
                independent_first_chunk=self.independent_first_chunk,
                sink=self.sink,
                window_size=self.window_size,
                video_temporal_mapping=self.video_temporal_mapping,
            ),
        }

    def _finalize_video_ref_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Finalize a decoded candidate before its packing budget is committed."""
        return sample

    def _duration(self, path: Path) -> Fraction:
        if path not in self._video_duration_cache:
            self._video_duration_cache[path] = _probe_video_duration(path)
        return self._video_duration_cache[path]

    def _decode_target_video(self, path: Path, start_frame: int) -> torch.Tensor:
        return _decode_video_on_frame_grid(
            path,
            start_frame=start_frame,
            num_frames=self.num_frames,
            fps=self.fps,
            height=self.height,
            width=self.width,
            spatial_filter=(
                f"scale={self.width}:{self.height}:force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={self.width}:{self.height}"
            ),
            description=f"target video from {path}",
        )

    def _decode_reference_video(self, path: Path, start_frame: int) -> torch.Tensor:
        return decode_reference_video(
            path,
            start_frame=start_frame,
            num_frames=self.num_frames,
            fps=self.fps,
            height=self.reference_height,
            width=self.reference_width,
        )

    def _decode_target_audio(self, path: Path, start_frame: int) -> torch.Tensor:
        assert isinstance(start_frame, int) and not isinstance(start_frame, bool) and start_frame >= 0
        target_samples = self.audio_t * _AUDIO_HOP_LENGTH
        start_sample = round(Fraction(start_frame * _AUDIO_SAMPLE_RATE, self.fps))
        filters = (
            f"aresample={_AUDIO_SAMPLE_RATE},"
            f"atrim=start_sample={start_sample}:end_sample={start_sample + target_samples},"
            "asetpts=PTS-STARTPTS"
        )
        payload = _run_ffmpeg(
            [
                "ffmpeg",
                "-v",
                "error",
                "-threads",
                "1",
                "-filter_threads",
                "1",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-af",
                filters,
                "-ac",
                "2",
                "-ar",
                str(_AUDIO_SAMPLE_RATE),
                "-c:a",
                "pcm_f32le",
                "-f",
                "f32le",
                "pipe:1",
            ],
            f"target audio from {path}",
        )
        sample_bytes = 2 * torch.tensor([], dtype=torch.float32).element_size()
        if len(payload) % sample_bytes:
            raise RuntimeError(
                f"FFmpeg decoded a partial stereo float32 sample from {path}"
            )
        waveform = torch.zeros((2, target_samples), dtype=torch.float32)
        if payload:
            decoded = torch.frombuffer(bytearray(payload), dtype=torch.float32).reshape(
                -1, 2
            )
            copy_samples = min(target_samples, decoded.shape[0])
            waveform[:, :copy_samples] = decoded[:copy_samples].T
        return waveform.contiguous()

    def _load_raw_media(
        self, entry: dict[str, Path], rng: random.Random
    ) -> dict[str, torch.Tensor]:
        video_path = entry["video_path"]
        reference_path = entry["reference_path"]
        available_duration = min(
            self._duration(video_path), self._duration(reference_path)
        )
        window_duration = Fraction(self.num_frames, self.fps)
        if available_duration < window_duration:
            raise ValueError(
                f"paired videos provide {available_duration}s but "
                f"{window_duration}s are required"
            )
        available_frames = math.floor(available_duration * self.fps)
        max_start_frame = max(0, available_frames - self.num_frames)
        start_frame = rng.randrange(max_start_frame + 1)
        return {
            "video_pixels": self._decode_target_video(video_path, start_frame),
            "reference_video_pixels": self._decode_reference_video(
                reference_path, start_frame
            ),
            "audio_waveform": self._decode_target_audio(video_path, start_frame),
        }

    def __iter__(self):
        worker_info = get_worker_info()
        physical_worker_id = worker_info.id if worker_info else 0
        physical_worker_count = worker_info.num_workers if worker_info else 1
        effective_workers = self.resume_context.num_workers or 1
        if physical_worker_count != effective_workers:
            raise ValueError(
                "worker topology mismatch: context expects "
                f"{effective_workers}, runtime has {physical_worker_count}"
            )
        logical_worker_id = (
            physical_worker_id + self.resume_context.next_logical_worker_id
        ) % physical_worker_count
        if logical_worker_id in self._decoded_worker_states:
            offset, avg_seqlen, cnt = self._decoded_worker_states[logical_worker_id]
        else:
            offset, avg_seqlen, cnt = self._initial_worker_state(logical_worker_id)

        while True:
            rng = random.Random(offset)
            samples: list[dict[str, Any]] = []
            cur_rows = 0
            num_retries = 0
            while len(samples) == 0 or cur_rows + avg_seqlen <= self.max_seqlen:
                entry = self.entries[rng.randrange(len(self.entries))]
                prompt = self._prompt_cache[entry["prompt_path"]]
                if self.text_dropout > 0.0 and rng.random() < self.text_dropout:
                    prompt = ""
                candidate = self._pack_video_ref_sample(prompt=prompt)
                packing_rows = int(candidate["packing_rows"])
                if (
                    self.max_seqlen_per_sample is not None
                    and packing_rows > self.max_seqlen_per_sample
                ) or cur_rows + packing_rows > self.max_seqlen:
                    num_retries += 1
                    if num_retries >= self.max_retries:
                        break
                    continue
                candidate.update(self._load_raw_media(entry, rng))
                candidate = self._finalize_video_ref_sample(candidate)
                packing_rows = int(candidate["packing_rows"])
                if (
                    self.max_seqlen_per_sample is not None
                    and packing_rows > self.max_seqlen_per_sample
                ) or cur_rows + packing_rows > self.max_seqlen:
                    num_retries += 1
                    if num_retries >= self.max_retries:
                        break
                    continue
                avg_seqlen = avg_seqlen * cnt / (cnt + 1) + packing_rows / (cnt + 1)
                cnt += 1
                cur_rows += packing_rows
                num_retries = 0
                samples.append(candidate)

            if not samples:
                raise ValueError("no raw video-reference atom fits max_seqlen")
            offset = yield_seed(offset)
            batch = {key: [sample[key] for sample in samples] for key in samples[0]}
            state_after = self._encode_worker_state(
                logical_worker_id, offset, avg_seqlen, cnt
            )
            yield WorkerStateEnvelope(batch, logical_worker_id, state_after)


class CausalVideoRefRawDiffusionForcingT2AVDataset(CausalVideoRefRawT2AVDataset):
    """Diffusion-forcing raw video-reference dataset."""

    _STATE_SCHEMA = "minimax_h3_videoref_causal_video_ref_raw_df_worker"
    forcing = "diffusion"


class CausalVideoRefRawTeacherForcingT2AVDataset(CausalVideoRefRawT2AVDataset):
    """Teacher-forcing raw video-reference dataset."""

    _STATE_SCHEMA = "minimax_h3_videoref_causal_video_ref_raw_tf_worker"
    forcing = "teacher"


EntryClass = CausalVideoRefRawDiffusionForcingT2AVDataset

__all__ = [
    "CausalVideoRefRawDiffusionForcingT2AVDataset",
    "CausalVideoRefRawT2AVDataset",
    "CausalVideoRefRawTeacherForcingT2AVDataset",
    "EntryClass",
    "decode_reference_video",
]
