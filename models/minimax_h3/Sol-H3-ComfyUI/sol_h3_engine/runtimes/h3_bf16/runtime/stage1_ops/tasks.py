"""Native task selection and input identity checks; no eager GPU imports."""
def task_of(case):
    task = case.get("task", "t2va")
    if task not in ("t2va", "fl2va", "ref2va"):
        raise ValueError(f"unsupported H3 task: {task}")
    return task


def ref_lora_alpha(rank, declared_alpha):
    """Preserve the Ref2VA checkpoint's alpha8/rank128, not native default1."""
    if rank != 128 or declared_alpha not in (None, 8):
        raise RuntimeError("Ref2VA adapter must retain native rank128/alpha8")
    return 8


def input_spec(case):
    return {"task": task_of(case), "first_frame": case.get("first_frame"),
            "last_frame": case.get("last_frame"), "references": case.get("references", [])}


def native_inputs(case):
    from fastvideo.api import InputConfig
    task = task_of(case)
    if task == "ref2va":
        from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3Reference
        return InputConfig(references=[MiniMaxH3Reference(source=item["path"], media_type=item["type"])
                                       for item in case["references"]])
    last = None
    if case.get("last_frame"):
        from PIL import Image
        with Image.open(case["last_frame"]) as source:
            last = source.copy()  # Native preparation owns EXIF/RGB/canvas processing.
    return InputConfig(image_path=case.get("first_frame"), last_image=last)


def validate_prepared_media(case, payload, batch):
    """Compare native prepared geometry/order with the external Qwen receipt.

    CPU preprocessing remains native. Do not duplicate full-video hashing or
    decode work here; request identity and the exact helper algorithms bind it.
    """
    task = task_of(case)
    if task == "t2va":
        if batch.extra.get("minimax_h3_keyframes") or batch.references:
            raise RuntimeError("T2VA unexpectedly contains visual conditions")
        return
    if payload.get("input_spec") != input_spec(case):
        raise RuntimeError("Qwen input order or media identity differs from Stage1")
    expected = payload.get("prepared_media")
    if not isinstance(expected, list):
        raise RuntimeError("conditioned Qwen payload requires prepared media geometry")
    actual = []
    if task == "fl2va":
        images = batch.extra.get("minimax_h3_keyframes", [])
        sources = [case[key] for key in ("first_frame", "last_frame") if case.get(key)]
        if len(images) != len(sources):
            raise RuntimeError("native first/last condition count differs")
        for path, image in zip(sources, images):
            actual.append({"type": "image", "path": path, "shape": [image.height, image.width, 3]})
    else:
        references = list(batch.references or [])
        if len(references) != len(case["references"]):
            raise RuntimeError("native reference count differs")
        for source, reference in zip(case["references"], references):
            item = {"type": reference.media_type, "path": source["path"], "has_audio": reference.has_audio}
            if reference.media_type == "image":
                item["shape"] = [reference.image.height, reference.image.width, 3]
            elif reference.media_type == "video":
                item["shape"] = list(reference.frames.shape)
            actual.append(item)
    if len(actual) != len(expected) or any(
            any(record.get(key) != value for key, value in item.items())
            for item, record in zip(actual, expected)):
        raise RuntimeError(f"Qwen/native media preprocessing differs: {actual}")
