# SPDX-License-Identifier: Apache-2.0
"""CPU media preparation for the pinned native Comfy MiniMax tokenizer.

Geometry, frame resampling and keyframe cropping are narrowly adapted from
hao-ai-lab/FastVideo at 3d8ac9d14bd697a89ede8f170cbfbca012a9edcc:
fastvideo/pipelines/basic/minimax_h3/{packing,reference}.py and
fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_input_preparation.py
(Apache-2.0).
Keep these pixels aligned with Stage1's native input preparation. Comfy owns
the vision processor, token presentation, temporal padding and token tags.
"""
from __future__ import annotations

import math


def input_spec(case):
    """Canonical identity shared by the conditioning payload and Stage1."""
    return {"task": case.get("task", "t2va"),
            "first_frame": case.get("first_frame"),
            "last_frame": case.get("last_frame"),
            "references": [dict(item) for item in case.get("references", [])]}


def load_rgb(path):
    from PIL import Image, ImageOps
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def prepare_keyframe_image(image, height, width, stretch):
    from PIL import Image
    if image.size == (width, height):
        return image
    if stretch:
        return image.resize((width, height), Image.Resampling.LANCZOS)
    scale = max(width / image.size[0], height / image.size[1])
    resized_size = (max(width, round(image.size[0] * scale)),
                    max(height, round(image.size[1] * scale)))
    left = max(0, (resized_size[0] - width) // 2)
    top = max(0, (resized_size[1] - height) // 2)
    resized = image.resize(resized_size, Image.Resampling.LANCZOS)
    return resized.crop((left, top, left + width, top + height))


def resolve_reference_image_size(width, height):
    if width <= 0 or height <= 0:
        raise ValueError("reference image dimensions must be positive")
    if not 1 / 4 <= width / height <= 4:
        raise ValueError("reference image aspect ratio must be between 1:4 and 4:1")
    scale = 2048 / min(width, height)
    return max(32, round(height * scale / 32) * 32), max(32, round(width * scale / 32) * 32)


def prepare_reference_image(image, height, width):
    from PIL import Image
    if image.size == (width, height):
        return image
    return image.resize((width, height), Image.Resampling.LANCZOS)


def resolve_canvas_size(aspect_width, aspect_height):
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError("reference video dimensions must be positive")
    ratio = aspect_width / aspect_height
    if not 1 / 4 <= ratio <= 4:
        raise ValueError("reference video aspect ratio must be between 1:4 and 4:1")
    if ratio >= 1:
        width, height = 768 * ratio, 768.0
    else:
        width, height = 768.0, 768 / ratio
    area = width * height
    if area > 768 * 1344:
        scale = (768 * 1344 / area) ** 0.5
        width, height = width * scale, height * scale
    return max(32, round(height / 32) * 32), max(32, round(width / 32) * 32)


def _validate_frames(frames):
    if frames.ndim != 4 or frames.shape[0] == 0 or frames.shape[-1] != 3:
        raise ValueError("reference video must contain nonempty RGB THWC frames")


def resample_reference_frames(frames, fps):
    import numpy as np
    _validate_frames(frames)
    if fps <= 0:
        raise ValueError("reference video frame rate must be positive")
    if fps == 24:
        return frames
    scale = 24 / fps
    slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(np.int64)
    repeats = np.diff(slots, append=math.floor(frames.shape[0] * scale + 0.5))
    return np.repeat(frames, repeats, axis=0)


def prepare_reference_frames(frames, num_frames=124):
    import numpy as np
    from PIL import Image
    _validate_frames(frames)
    frames = frames[:num_frames]
    height, width = resolve_canvas_size(frames.shape[2], frames.shape[1])
    if frames.shape[1:3] == (height, width):
        return frames
    return np.stack([np.asarray(Image.fromarray(frame).resize(
        (width, height), Image.Resampling.LANCZOS)) for frame in frames])


def reference_video_samples(frame_count):
    """Return native 2-fps indices and *per-frame* Comfy timestamps.

    FastVideo averages timestamp pairs for each temporal patch. Comfy performs
    that same averaging and odd-last-frame duplication inside its tokenizer;
    passing block timestamps or padding here would apply that policy twice.
    """
    indices, cursor = [], 0.0
    while round(cursor) < frame_count:
        index = round(cursor)
        if not indices or index > indices[-1]:
            indices.append(index)
        cursor += 24 / 2.0
    return indices, [index / 2.0 for index in range(len(indices))]


def decode_reference_video(path):
    """Native RGB frame/rotation policy; Qwen only needs audio presence.

    Stage1 decodes the actual soundtrack with its audio VAE contract. No audio
    samples are input to this text/vision encoder, including standalone audio.
    """
    import av
    import numpy as np
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("reference video has no video stream")
        stream = container.streams.video[0]
        rate = stream.average_rate or getattr(stream, "guessed_rate", None)
        if rate is None:
            raise ValueError("reference video frame rate is unavailable")
        max_frames = math.ceil(15.0 * float(rate)) + 1
        frames, rotation = [], 0.0
        for frame in container.decode(stream):
            rotation = float(getattr(frame, "rotation", 0.0) or 0.0)
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= max_frames:
                break
        if not frames:
            raise ValueError("reference video has no decoded frames")
        has_audio = bool(container.streams.audio)
    result = np.stack(frames)
    turns = round(rotation / 90.0) % 4
    if turns:
        result = np.ascontiguousarray(np.rot90(result, k=-turns, axes=(1, 2)))
    return result, float(rate), has_audio


def prepare_inputs(case, torch):
    """Build native ``clip.tokenize`` kwargs and JSON-safe preparation facts."""
    task = case.get("task", "t2va")
    if task == "t2va":
        return {"images": []}, []
    if task not in ("fl2va", "ref2va"):
        raise ValueError(f"unknown conditioning task: {task}")
    import numpy as np

    def image_tensor(image):
        return torch.from_numpy(np.array(image).astype(np.float32)[None] / 255.0)

    prepared = []
    if task == "fl2va":
        images = []
        for anchor, key in (("first", "first_frame"), ("last", "last_frame")):
            path = case.get(key)
            if path is None:
                continue
            # Native FastVideo stretches the first *existing* keyframe, even
            # when it is last-only; subsequent keyframes are center-cropped.
            image = prepare_keyframe_image(load_rgb(path), 384, 672, stretch=not images)
            images.append(image_tensor(image))
            prepared.append({"type": "image", "path": path, "anchor": anchor,
                             "shape": [384, 672, 3], "has_audio": False})
        if not images:
            raise ValueError("fl2va requires a first or last frame")
        return {"images": images}, prepared

    items = []
    for reference in case["references"]:
        kind, path = reference["type"], reference["path"]
        if kind == "image":
            image = load_rgb(path)
            height, width = resolve_reference_image_size(*image.size)
            image = prepare_reference_image(image, height, width)
            items.append({"type": "image", "data": image_tensor(image)})
            prepared.append({"type": kind, "path": path,
                             "shape": [height, width, 3], "has_audio": False})
        elif kind == "video":
            frames, fps, has_audio = decode_reference_video(path)
            frames = prepare_reference_frames(resample_reference_frames(frames, fps))
            indices, timestamps = reference_video_samples(len(frames))
            if has_audio:
                items.append({"type": "audio"})
            items.append({"type": "video", "data": torch.from_numpy(
                frames[indices].astype(np.float32) / 255.0), "timestamps": timestamps})
            prepared.append({"type": kind, "path": path, "shape": list(frames.shape),
                             "has_audio": has_audio, "sampled_indices": indices,
                             "timestamps": timestamps})
        elif kind == "audio":
            items.append({"type": "audio"})
            prepared.append({"type": kind, "path": path, "shape": None, "has_audio": True})
        else:
            raise ValueError(f"unsupported reference type: {kind}")
    return {"minimax_ref_items": items}, prepared
