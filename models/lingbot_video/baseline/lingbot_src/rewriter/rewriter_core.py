import io
import json
import re

import requests
from PIL import Image

try:
    from json_repair import repair_json
except ImportError:
    repair_json = None

from system_prompts import (
    VIDEO_STEP1_EXPAND, VIDEO_STEP2_MAP, IMAGE_STEP1_EXPAND, IMAGE_STEP2_MAP,
)

# mode -> (step1 prompt, step2 prompt, feed image?, add duration?)
MODES = {
    "t2v":  dict(s1=VIDEO_STEP1_EXPAND, s2=VIDEO_STEP2_MAP, image=False, duration=True),
    "ti2v": dict(s1=VIDEO_STEP1_EXPAND, s2=VIDEO_STEP2_MAP, image=True,  duration=True),
    "t2i":  dict(s1=IMAGE_STEP1_EXPAND, s2=IMAGE_STEP2_MAP, image=False, duration=False),
}


def _has_cjk(s):
    return any("一" <= c <= "鿿" for c in s)


def load_image(src):
    """Load a first-frame image from a local path / http(s) URL / PIL.Image. Returns RGB PIL."""
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    if isinstance(src, str) and re.match(r"^https?://", src):
        return Image.open(io.BytesIO(requests.get(src, timeout=30).content)).convert("RGB")
    return Image.open(src).convert("RGB")


def _step1_text(mode, prompt, dur):
    sys = MODES[mode]["s1"]
    if mode == "t2i":
        return sys + "\n\nUser image prompt:\n" + prompt
    dur_line = f"\n\n视频时长：{dur} 秒" if _has_cjk(prompt) else f"\n\nVideo Duration: {dur} seconds"
    return sys + "\n\n" + prompt + dur_line


def _step2_text(mode, detailed, dur):
    sys = MODES[mode]["s2"]
    if mode == "t2i":
        return sys + "\n\nDETAILED CAPTION:\n" + detailed
    return (sys + f"\n\nVideo Duration: {dur} seconds\n\nDETAILED CAPTION:\n"
            + detailed + "\n\nOutput the JSON now.")


def parse_json(raw):
    """Parse the step2 output. VLMs are occasionally unstable (missing quotes, trailing
    commas, ``` fences), so we strip any code fence and then run json_repair."""
    if repair_json is None:
        raise ImportError("rewriter parsing requires the json_repair package.")
    s = (raw or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", s, re.DOTALL)
    if m:
        s = m.group(1)
    try:
        obj = repair_json(s, return_objects=True)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def save_caption(result, duration, path):
    """Save as a JSON file: ``{"caption": <structured JSON caption>, "duration": <seconds>}``.

    ``duration`` is the integer seconds for T2V/TI2V and ``null`` for T2I (a still image
    has no duration). This is exactly the structured-prompt JSON the DiT runner consumes.
    """
    dur = int(round(duration)) if MODES[result["mode"]]["duration"] else None
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"caption": result["json"], "duration": dur}, f, ensure_ascii=False, indent=2)
    return path


class Rewriter:
    """Two-stage orchestrator. The backend implements ``generate(text, image, use_lora) -> str``."""

    def __init__(self, backend):
        self.backend = backend

    def rewrite(self, prompt, mode="t2v", first_frame=None, duration=5):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}")
        cfg = MODES[mode]
        dur = int(round(duration))
        img = None
        if cfg["image"]:
            if first_frame is None:
                raise ValueError(f"{mode} requires first_frame (path / URL / PIL.Image)")
            img = load_image(first_frame)
        # step1 EXPAND -- base model (no LoRA)
        detailed = self.backend.generate(_step1_text(mode, prompt, dur), img, use_lora=False).strip()
        # step2 MAP -- base + LoRA
        raw = self.backend.generate(_step2_text(mode, detailed, dur), img, use_lora=True).strip()
        return {"mode": mode, "detailed": detailed, "json": parse_json(raw), "json_raw": raw}
