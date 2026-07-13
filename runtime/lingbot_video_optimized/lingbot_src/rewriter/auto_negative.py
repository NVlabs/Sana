import argparse
import copy
import json
import os
import re

from inference import make_backend
from rewriter_core import load_image

try:
    from json_repair import repair_json
except ImportError:
    repair_json = None

# Deliberately broad defaults. Auto-negative keeps a delete-only subset per sample.
DEFAULT_NEGATIVE_VIDEO = {
    "universal_negative": {
        "visual_quality": [
            "low quality", "worst quality", "blurry", "pixelated", "jpeg artifacts", "low resolution",
            "unstable color", "color flicker",
            "underexposed", "overexposed", "washed out", "over-saturation", "poor color balance",
            "invisible subject", "subject hidden in darkness", "crushed blacks", "blown-out highlights",
            "macroblocking", "rolling shutter", "temporal aliasing", "color banding", "edge halos",
            "chromatic aberration", "moiré patterns", "high-frequency noise", "grainy texture", "low frame rate",
        ],
        "lighting_failure": [
            "harsh flat lighting", "uniformly lit without variation", "no atmosphere or depth",
            "inconsistent light direction", "hard-edged unrealistic shadows",
        ],
        "physical_plausibility": [
            "physically implausible motion", "objects defying gravity", "violates real-world physics",
            "impossible contact or penetration", "nonsensical interactions", "illogical scene",
        ],
        "artistic_style": [
            "painting", "illustration", "drawing", "cartoon", "3d render", "cgi", "sketch", "digital art",
        ],
        "composition_and_content": [
            "text", "watermark", "signature", "logo", "subtitles",
            "pillarboxed", "side bars", "portrait image in landscape frame",
        ],
        "temporal_and_motion_stability": [
            "flickering", "jittery", "motion blur", "temporal inconsistency", "warping", "morphing",
            "incoherent motion", "unnatural movement", "static object with sudden jump",
            "frame-to-frame inconsistency", "shaky footage", "jump cuts", "hard cut",
            "unnatural scene transitions",
        ],
        "material_and_structure": [
            "plastic-like glass", "unrealistic texture", "deformed bottle",
            "liquid freezing improperly", "distorted reflections",
        ],
    }
}

DEFAULT_NEGATIVE_IMAGE = {
    "universal_negative": {
        "visual_quality": [
            "low quality", "worst quality", "blurry", "pixelated", "jpeg artifacts", "low resolution",
            "underexposed", "overexposed", "washed out", "over-saturation", "poor color balance",
            "invisible subject", "subject hidden in darkness", "crushed blacks", "blown-out highlights",
            "color banding", "edge halos", "chromatic aberration", "moiré patterns",
            "high-frequency noise", "grainy texture",
        ],
        "lighting_failure": [
            "harsh flat lighting", "uniformly lit without variation", "no atmosphere or depth",
            "inconsistent light direction", "hard-edged unrealistic shadows",
        ],
        "physical_plausibility": [
            "objects defying gravity", "violates real-world physics",
            "impossible contact or penetration", "nonsensical interactions", "illogical scene",
        ],
        "artistic_style": [
            "painting", "illustration", "drawing", "cartoon", "3d render", "cgi", "sketch", "digital art",
        ],
        "composition_and_content": [
            "text", "watermark", "signature", "logo",
            "pillarboxed", "side bars", "portrait image in landscape frame",
        ],
        "material_and_structure": [
            "plastic-like glass", "unrealistic texture", "deformed bottle",
            "distorted reflections",
        ],
    }
}


def default_negative_for(mode):
    """t2i -> still-image default; t2v / ti2v -> video default (ti2v output is a video)."""
    return DEFAULT_NEGATIVE_IMAGE if mode == "t2i" else DEFAULT_NEGATIVE_VIDEO

NEG_SYS = r"""You build a per-sample negative prompt for a video / image generation model.

You are given (1) the INTENDED content -- a structured caption describing the video / image the user wants
(and optionally its first frame) -- and (2) a DEFAULT NEGATIVE: a deliberately over-complete JSON list of
generic defects and failure modes, written to cover many scenarios at once, so it INTENTIONALLY contains
mutually contradictory terms (e.g. both "static object with sudden jump" and "motion blur", both
"underexposed" and "overexposed").

A negative prompt is a CFG anchor the model is pushed AWAY from. So it must NEVER contain anything the
intended content legitimately wants, or the model is pushed away from what the user asked for.

TASK: produce a customized negative for THIS sample by DELETING, from the default negative, any term that
CONTRADICTS or would SUPPRESS something the intended content legitimately wants. Keep everything else.

RULES:
- DELETE ONLY. Keep the surviving terms verbatim, in the SAME JSON structure. Never add, reword, or merge.
- Delete a term only when it clearly clashes with the intended content. When in doubt, KEEP it -- the
  defaults are safe generic defects.
- Judge EVERY category against the intent. Examples (not exhaustive -- reason about the actual sample):
  * wants shot cuts / montage / transitions        -> remove "jump cuts", "hard cut", "unnatural scene transitions".
  * an intentionally static / still scene           -> remove "static object with sudden jump".
  * intentional motion blur / long exposure / timelapse -> remove "motion blur".
  * deliberately dark / moody / night / deep-shadow lighting -> remove "underexposed",
    "subject hidden in darkness", and "crushed blacks"; keep "invisible subject" unless the
    subject is meant to be invisible.
  * desired blurred background / speed blur / long exposure look -> remove "motion blur".
  * fantasy / surreal / dreamlike / magic / zero-gravity / physics-bending by design -> remove only
    the physical_plausibility terms that would fight the intended content.
  * a painting / illustration / cartoon / CGI / sketch by design -> remove the matching artistic_style terms.
  * flickering neon / strobe / candlelight by design -> remove "flickering", "color flicker".
- WHOLE-BLOCK removal is RARE and requires explicit evidence in the intended content:
  * Remove the ENTIRE physical_plausibility category ONLY when the caption explicitly requests
    fantasy, surreal, dreamlike, magical, zero-gravity, weightless, or physics-bending content.
    Do NOT remove this category for normal live-action, real-world people, animals, vehicles,
    sports, stunts, dust, speed, impact, or camera-captured scenes.
  * Remove the ENTIRE artistic_style category ONLY when the caption explicitly requests a
    non-photoreal style such as painting, illustration, cartoon, drawing, sketch, CGI, 3D render,
    digital art, anime, or stylized animation. Do NOT remove this category for ordinary cinematic,
    dramatic, moody, high-tech, sci-fi, or camera-shot live-action scenes.
- NEVER delete generic defects that no legitimate content wants: "low quality", "worst quality", "blurry",
  "jpeg artifacts", "low resolution", "pixelated", "text", "watermark", "logo", "subtitles",
  "pillarboxed", "side bars", "warping", "morphing".

OUTPUT: return ONLY a valid JSON object in the SAME structure as the default negative (the same
"universal_negative" object, same category keys), with the contradicting terms removed from their category
arrays. It MUST be strict parseable JSON -- double quotes, no trailing commas, no comments, no code fence,
and no text before or after the JSON object."""


def _parse(raw):
    if repair_json is None:
        raise ImportError("auto negative requires the json_repair package.")
    s = (raw or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", s, re.DOTALL)
    if m:
        s = m.group(1)
    try:
        obj = repair_json(s, return_objects=True)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _enforce_delete_only(pruned, default):
    """Keep only terms that exist in the mode's default (in default order). Drops any term the model
    added or reworded, guaranteeing the result is a strict subset of the default."""
    if not isinstance(pruned, dict):
        return copy.deepcopy(default)
    pn = pruned.get("universal_negative", {})
    if not isinstance(pn, dict):
        return copy.deepcopy(default)
    out = {"universal_negative": {}}
    for cat, arr in default["universal_negative"].items():
        if cat not in pn or not isinstance(pn.get(cat), list):
            out["universal_negative"][cat] = list(arr)
            continue
        kept_set = set(pn.get(cat) or [])
        out["universal_negative"][cat] = [t for t in arr if t in kept_set]
    return out


def _caption_text(caption):
    return caption if isinstance(caption, str) else json.dumps(caption, ensure_ascii=False)


_PHYSICAL_BLOCK_HINTS = (
    "fantasy", "surreal", "dreamlike", "dream-like", "magic", "magical", "supernatural",
    "physics-bending", "physics bending", "impossible physics", "anti-gravity",
    "antigravity", "zero gravity", "zero-gravity", "weightless", "weightlessness",
    "floating in space", "outer space", "astronaut",
)

_STYLE_BLOCK_HINTS = (
    "painting", "illustration", "cartoon", "drawing", "sketch", "cgi", "3d render",
    "3d-render", "digital art", "anime", "stylized animation", "claymation",
    "stop motion", "stop-motion",
)

_DARK_SCENE_HINTS = (
    "dark", "dim", "low light", "low-light", "night", "nighttime", "moody",
    "gloomy", "ominous", "deep shadow", "deep shadows", "dark shadows",
    "dramatic and moody",
)

_MOTION_BLUR_HINTS = (
    "motion blur", "motion-blur", "blurred background", "background is blurred",
    "background appears blurred", "blurred landscape", "blurred scenery",
    "blurred surroundings", "blurred by", "speed blur", "long exposure",
)


def _has_any_hint(text, hints):
    lowered = text.lower()
    return any(hint in lowered for hint in hints)


def _delete_terms(category_terms, terms_to_delete):
    delete_set = set(terms_to_delete)
    return [term for term in category_terms if term not in delete_set]


def _restore_unjustified_block_deletions(pruned, default, caption):
    out = copy.deepcopy(pruned)
    text = _caption_text(caption)
    categories = out["universal_negative"]
    default_categories = default["universal_negative"]
    if not categories.get("physical_plausibility") and not _has_any_hint(text, _PHYSICAL_BLOCK_HINTS):
        categories["physical_plausibility"] = list(default_categories["physical_plausibility"])
    if not categories.get("artistic_style") and not _has_any_hint(text, _STYLE_BLOCK_HINTS):
        categories["artistic_style"] = list(default_categories["artistic_style"])
    return out


def _apply_caption_pruning_hints(pruned, caption):
    out = copy.deepcopy(pruned)
    text = _caption_text(caption)
    categories = out["universal_negative"]
    if _has_any_hint(text, _DARK_SCENE_HINTS):
        categories["visual_quality"] = _delete_terms(
            categories["visual_quality"],
            ("underexposed", "subject hidden in darkness", "crushed blacks"),
        )
    if _has_any_hint(text, _MOTION_BLUR_HINTS) and "temporal_and_motion_stability" in categories:
        categories["temporal_and_motion_stability"] = _delete_terms(
            categories["temporal_and_motion_stability"],
            ("motion blur",),
        )
    return out


def _removed(pruned, default):
    removed = []
    pn = pruned["universal_negative"]
    for cat, arr in default["universal_negative"].items():
        for t in arr:
            if t not in pn[cat]:
                removed.append(t)
    return removed


def build_prompt(caption, mode, default):
    cap = caption if isinstance(caption, str) else json.dumps(caption, ensure_ascii=False)
    return (NEG_SYS
            + f"\n\n## MODE: {mode}"
            + "\n\n## INTENDED CONTENT (structured caption):\n```json\n" + cap + "\n```"
            + "\n\n## DEFAULT NEGATIVE (delete the contradicting terms, keep the rest):\n```json\n"
            + json.dumps(default, ensure_ascii=False) + "\n```"
            + "\n\nOutput ONLY the edited negative JSON now.")


class NegativePromptEditor:
    """Prune the mode's default negative per sample. Backend implements
    ``generate(text, image, use_lora=False) -> str`` (zero-shot base model)."""

    def __init__(self, backend):
        self.backend = backend

    def edit(self, caption, mode="t2v", first_frame=None):
        default = default_negative_for(mode)
        img = load_image(first_frame) if first_frame is not None else None
        raw = self.backend.generate(build_prompt(caption, mode, default), img, use_lora=False)
        parsed = _parse(raw)
        pruned = _enforce_delete_only(parsed, default)
        if parsed is not None:
            pruned = _restore_unjustified_block_deletions(pruned, default, caption)
            pruned = _apply_caption_pruning_hints(pruned, caption)
        return {
            "negative_json": pruned,
            "negative_str": json.dumps(pruned, ensure_ascii=False),
            "removed": _removed(pruned, default),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="transformers", choices=["transformers"],
                    help="rewriter backend; only the in-process transformers backend is bundled")
    ap.add_argument("--mode", default="t2v", choices=["t2v", "ti2v", "t2i"])
    ap.add_argument("--caption", required=True, help="rewritten structured caption: JSON file path or inline JSON")
    ap.add_argument("--first-frame", default=None, help="optional first frame (path / URL) for ti2v / t2i")
    ap.add_argument("--output", default=None, help="save the pruned negative JSON to this path")
    ap.add_argument("--base", default=os.environ.get("REWRITER_BASE_MODEL", ""))
    ap.add_argument("--adapter", default=os.environ.get("REWRITER_ADAPTER", ""))
    args = ap.parse_args()

    if os.path.isfile(args.caption):
        with open(args.caption, encoding="utf-8") as f:
            cap = json.load(f)
        cap = cap.get("caption", cap) if isinstance(cap, dict) else cap   # accept {caption, duration} too
    else:
        cap = args.caption

    editor = NegativePromptEditor(make_backend(args.backend, args.base, args.adapter))
    out = editor.edit(cap, args.mode, args.first_frame)

    print("========== REMOVED ==========\n" + json.dumps(out["removed"], ensure_ascii=False))
    print("\n========== NEGATIVE (pruned) ==========\n" + out["negative_str"])
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out["negative_json"], f, ensure_ascii=False, indent=2)
        print(f"\n[saved] {args.output}")


if __name__ == "__main__":
    main()
