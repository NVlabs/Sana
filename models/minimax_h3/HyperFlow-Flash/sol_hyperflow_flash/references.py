"""ref2va request files: a prompt plus an ordered list of references, read into diffusers `MiniMaxH3Reference` objects.

    {"name": "demo", "prompt": "...", "seed": 0,
     "references": [{"video": "clip.mp4"},                    # frames + the file's own soundtrack (<Video k> + <Audio j>)
                    {"audio": "voice.mp3"},                   # audio reference (needs an image or video reference too)
                    {"image": "subject.png"},                 # image reference
                    {"image": "clip.mp4", "frame": 48}]}      # one decoded frame of a video file as an image reference

Relative media paths resolve against `media_dir` (default: the request file's directory). The order is semantic: it numbers the
<Picture i> / <Video k> / <Audio j> labels of the prompt and lays the references out on the rotary clock.
"""
from __future__ import annotations

import json
from pathlib import Path

KINDS = ("image", "video", "audio")


def parse_request(path, media_dir=None):
    """Validated request dict with absolute media paths (no media decoded, no diffusers import)."""
    path = Path(path)
    req = json.loads(path.read_text(encoding="utf-8"))
    base = Path(media_dir) if media_dir else path.parent
    if not isinstance(req.get("prompt"), str) or not req["prompt"].strip():
        raise ValueError(f"{path}: 'prompt' must be a non-empty string")
    refs = req.get("references")
    if not isinstance(refs, list) or not refs:
        raise ValueError(f"{path}: 'references' must be a non-empty list")
    out = []
    for i, r in enumerate(refs):
        kinds = [k for k in KINDS if k in r]
        if len(kinds) != 1:
            raise ValueError(f"{path}: references[{i}] needs exactly one of {KINDS}, got {sorted(r)}")
        kind = kinds[0]
        if "frame" in r and kind != "image":
            raise ValueError(f"{path}: references[{i}]: 'frame' only selects a frame for an image reference")
        media = Path(r[kind])
        media = media if media.is_absolute() else base / media
        out.append(dict(kind=kind, path=str(media), frame=None if "frame" not in r else int(r["frame"])))
    n = {k: sum(x["kind"] == k for x in out) for k in KINDS}
    if n["image"] > 9 or n["video"] > 3 or n["audio"] > 3 or len(out) > 12:
        raise ValueError(f"{path}: MiniMax-H3 takes at most 9 images, 3 videos, 3 audio clips, 12 references ({n})")
    if n["audio"] and not (n["image"] or n["video"]):
        raise ValueError(f"{path}: an audio reference needs an image or video reference too")
    return dict(name=req.get("name", path.stem), prompt=req["prompt"].strip(), seed=req.get("seed"), references=out,
                source=req.get("source"))


def decode_frame(path, index):
    """Frame `index` of a video file as a uint8 [H, W, 3] array (PyAV)."""
    import av

    with av.open(str(path)) as c:
        for j, frame in enumerate(c.decode(c.streams.video[0])):
            if j == index:
                return frame.to_ndarray(format="rgb24")
    raise ValueError(f"{path} has no frame {index}")


def build_references(spec):
    """diffusers MiniMaxH3Reference objects for a parse_request() dict (decodes the media once, on the host)."""
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference

    refs = []
    for r in spec["references"]:
        if not Path(r["path"]).is_file():
            raise FileNotFoundError(r["path"])
        if r["kind"] == "image" and r["frame"] is not None:
            refs.append(MiniMaxH3Reference(image=decode_frame(r["path"], r["frame"])))
        else:
            refs.append(MiniMaxH3Reference(**{r["kind"]: r["path"]}))
    return refs


def describe(refs):
    """Per-reference summary for the provenance record."""
    out = []
    for r in refs:
        d = dict(kind=r.kind, has_audio=bool(r.has_audio))
        if r.kind == "image":
            img = r.image
            d["size"] = list(img.size) if hasattr(img, "size") and not hasattr(img, "shape") else list(img.shape)
        elif r.kind == "video":
            d.update(frames_shape=list(r.video.shape), fps=float(r.fps))
        if r.has_audio:
            d.update(audio_shape=list(r.audio.shape), sample_rate=r.sample_rate)
        out.append(d)
    return out


def resolve_seed(cli_seed, spec=None) -> int:
    """Seed precedence: the command line, then the request file's "seed", then 0."""
    if cli_seed is not None:
        return int(cli_seed)
    if spec is not None and spec.get("seed") is not None:
        return int(spec["seed"])
    return 0
