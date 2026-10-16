"""Small ComfyUI surface: profile, prompt and seed -> audio/video + timings."""
import json
import os
from pathlib import Path
import time

from . import bridge
from .config import RESOLUTIONS, DEFAULT_STEPS, load_profile, profile_resolutions, profile_modes, profile_names

MODES = ["text", "first_frame", "reference"]
MODE_LABELS = {"first_frame": "first-frame", "reference": "reference-image"}
BACKENDS = ["local"]
# Latest update per node for GET /solh3/progress (polling fallback for reloads).
PROGRESS = {}


def publish(node, prompt_id, stage, started, **fields):
    value = {"node": node, "prompt_id": prompt_id, "stage": stage, "step": None, "steps": None,
             "elapsed_s": round(time.monotonic() - started, 2), "message": ""}
    value.update(fields)
    PROGRESS[str(node)] = value
    try:
        from server import PromptServer
        PromptServer.instance.send_sync("solh3_progress", value)
    except Exception:
        pass  # No server (CPU tests) or no client: polling still sees PROGRESS.
    return value


def resolve_images(image):
    """Comma-separated uploaded names (ComfyUI input dir) -> absolute paths."""
    import folder_paths
    paths = []
    for name in (part.strip() for part in image.split(",")):
        if name:
            path = Path(folder_paths.get_annotated_filepath(name))
            if not path.is_file():
                raise ValueError("Uploaded image not found: %s" % name)
            paths.append(str(path.resolve()))
    return paths


def check_request(profile, mode, image, backend, resolution="768p"):
    if backend not in BACKENDS:
        raise ValueError("Unknown backend: %s" % backend)
    if mode not in MODES:
        raise ValueError("Unknown mode: %s" % mode)
    if resolution not in RESOLUTIONS:
        raise ValueError("Unknown resolution: %s" % resolution)
    if resolution not in profile_resolutions(profile):
        raise ValueError("This pipeline renders %s only" % ", ".join(profile_resolutions(profile)))
    if mode != "text":
        if mode not in profile_modes(profile):
            raise ValueError("This pipeline does not support %s mode" % MODE_LABELS[mode])
        if not image.strip():
            raise ValueError("Upload an image for %s mode" % MODE_LABELS[mode])
        if mode == "first_frame" and len([n for n in image.split(",") if n.strip()]) != 1:
            raise ValueError("First-frame mode takes exactly one image")


def write_sidecar(output_root, prompt_id, result, mode, seed, resolution="768p"):
    """Library record; SaveVideo writes the MP4, so the record lives beside the library."""
    folder = Path(output_root) / "SolH3" / "library"
    folder.mkdir(parents=True, exist_ok=True)
    record = {"id": prompt_id, "engine_output": result.get("output"),
              "prompt": result.get("prompt"), "mode": mode, "profile": result.get("profile"),
              "seed": seed, "resolution": resolution, "e2e_s": result.get("e2e_s"), "created": time.time()}
    temporary = folder / (prompt_id + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    os.replace(temporary, folder / (prompt_id + ".json"))


def current_prompt_id():
    try:
        from server import PromptServer
        return getattr(PromptServer.instance, "last_prompt_id", None)
    except Exception:
        return None


class SolH3Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "profile": (profile_names(),),
            "prompt": ("STRING", {"multiline": True, "default": "A bird glides over a mountain lake. overall_soundscape: Soft wind and distant birds."}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2**53 - 1}),
            "keep_loaded": ("BOOLEAN", {"default": True}),
        }, "optional": {
            # Optional with defaults so workflows saved before Studio still load.
            "mode": (MODES, {"default": "text"}),
            "image": ("STRING", {"default": ""}),
            "backend": (BACKENDS, {"default": "local"}),
            "resolution": (list(RESOLUTIONS), {"default": "768p"}),
        }, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video_with_audio", "timing_json")
    FUNCTION = "generate"
    CATEGORY = "Sol Engine"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Every queued request must run: never time a ComfyUI cache hit.
        return float("nan")

    def generate(self, profile, prompt, seed, keep_loaded=True, mode="text", image="",
                 backend="local", resolution="768p", unique_id=None):
        if not prompt.strip():
            raise ValueError("Enter a nonempty prompt")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Seed must be a nonnegative 63-bit integer")
        check_request(profile, mode, image or "", backend, resolution)
        import folder_paths
        import comfy.model_management as mm
        from comfy_api.latest import InputImpl
        images = resolve_images(image or "") if mode != "text" else []
        node, prompt_id, started = unique_id, current_prompt_id() or "", time.monotonic()
        publish(node, prompt_id, "queued", started)
        output_root = folder_paths.get_output_directory()
        try:
            settings = load_profile(profile)
            mm.unload_all_models()
            mm.soft_empty_cache()
            cold = not bridge.is_warm(settings, output_root)
            steps = settings.get("steps", DEFAULT_STEPS.get(settings["precision"]))
            publish(node, prompt_id, "loading" if cold else "conditioning", started, steps=steps)

            def progress(update):
                stage = update.get("stage", "generating")
                if stage == "generating" and cold:
                    stage = "loading"  # Without worker updates a cold start is mostly loading.
                publish(node, prompt_id, stage, started, step=update.get("step"),
                        steps=update.get("steps", steps), message=update.get("message", ""))

            try:
                result = bridge.generate(settings, output_root, prompt, seed,
                                         cancel=mm.throw_exception_if_processing_interrupted,
                                         mode=mode, images=images, progress=progress, resolution=resolution)
            finally:
                if not keep_loaded:
                    bridge.close()
            publish(node, prompt_id, "saving", started, steps=steps)
            if prompt_id:
                write_sidecar(output_root, prompt_id, result, mode, seed, resolution)
        except BaseException as error:
            publish(node, prompt_id, "error", started, message=str(error) or type(error).__name__)
            raise
        publish(node, prompt_id, "done", started, steps=steps, step=steps,
                message="%.1f s" % result.get("e2e_s", 0))
        report = json.dumps(result, indent=2, ensure_ascii=False)
        # SaveVideo owns the native player. Keeping one player avoids duplicate
        # preview sizing conflicts with a legacy node's timing widget.
        # Lets Studio show the engine MP4 at once (and in graphs without SaveVideo).
        relative = Path(result["output"]).resolve().relative_to(Path(output_root).resolve())
        video = {"filename": relative.name, "subfolder": relative.parent.as_posix(), "type": "output"}
        return {"ui": {"text": [report], "solh3_video": [video]},
                "result": (InputImpl.VideoFromFile(result["output"]), report)}


class SolH3Release:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"after": ("VIDEO",)}}

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "release"
    CATEGORY = "Sol Engine"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def release(self, after):
        bridge.close()
        return (after,)


NODE_CLASS_MAPPINGS = {"SolH3Generate": SolH3Generate, "SolH3Release": SolH3Release}
NODE_DISPLAY_NAME_MAPPINGS = {"SolH3Generate": "Sol-H3 · Generate Video",
                              "SolH3Release": "Sol-H3 · Release GPU"}
