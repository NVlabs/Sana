"""HTTP surface for the Studio (/solh3/*). Pure helpers stay importable without ComfyUI."""
import io
import json
from pathlib import Path

LIBRARY = Path("SolH3") / "library"
MATCH_WINDOW_S = 600  # SaveVideo writes its MP4 shortly after the node's sidecar.


def _sidecars(output_root):
    rows = []
    for path in (Path(output_root) / LIBRARY).glob("*.json"):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(record, dict) and isinstance(record.get("created"), (int, float)):
            rows.append(record)
    return sorted(rows, key=lambda row: row["created"])


def list_videos(output_root, limit=24, history=None):
    """Newest-first SolH3/**/*.mp4 merged with node sidecars.

    `history(prompt_id)` may return saved (subfolder, filename) pairs from ComfyUI's
    history for an exact match; otherwise a sidecar claims the first MP4 written after it.
    """
    root = Path(output_root).resolve()
    videos = []
    for path in (root / "SolH3").rglob("*.mp4"):
        if LIBRARY in path.relative_to(root).parents:
            continue
        videos.append((path.stat().st_mtime, path))
    videos.sort()
    meta, claimed, known = {}, set(), {path for _, path in videos}
    for record in _sidecars(root):
        exact = []
        if history is not None and record.get("id"):
            try:
                exact = [root / sub / name for sub, name in history(record["id"]) or ()]
            except Exception:
                exact = []  # History is an optimization; mtime matching still works.
        match = next((p for p in exact if p in known and p not in claimed), None)
        if match is None:
            match = next((p for mtime, p in videos if p not in claimed
                          and record["created"] - 2 <= mtime <= record["created"] + MATCH_WINDOW_S), None)
        if match is not None:
            claimed.add(match)
            meta[match] = record
    items = []
    for mtime, path in reversed(videos):
        record = meta.get(path, {})
        relative = path.relative_to(root)
        items.append({"id": relative.as_posix(), "file": path.name,
                      "subfolder": relative.parent.as_posix(), "type": "output",
                      "prompt_id": record.get("id"), "prompt": record.get("prompt"),
                      "mode": record.get("mode"), "profile": record.get("profile"),
                      "seed": record.get("seed"), "e2e_s": record.get("e2e_s"),
                      "created": record.get("created", mtime)})
        if len(items) >= limit:
            break
    return {"items": items}


def video_path(output_root, identity):
    root = Path(output_root).resolve()
    path = (root / str(identity or "")).resolve()
    path.relative_to(root / "SolH3")  # ValueError outside the Studio library.
    if path.suffix != ".mp4" or not path.is_file():
        raise ValueError("Saved video unavailable")
    return path


def thumbnail(output_root, identity):
    """First frame as JPEG, cached; needs PyAV (ComfyUI's video dependency)."""
    path = video_path(output_root, identity)
    cache = Path(output_root).resolve() / LIBRARY / "thumbs"
    saved = cache / (path.relative_to(Path(output_root).resolve() / "SolH3").as_posix()
                     .replace("/", "__") + ".jpg")
    if saved.is_file() and saved.stat().st_mtime >= path.stat().st_mtime:
        return saved.read_bytes()
    import av  # ImportError -> 404 at the route.
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        image = next(container.decode(stream)).to_image()
    image.thumbnail((480, 270))
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=82)
    data = buffer.getvalue()
    try:
        cache.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(data)
    except OSError:
        pass  # A read-only output directory still serves thumbnails.
    return data


GALLERY_MEDIA = {".mp4": "video/mp4", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                 ".webp": "image/webp"}


def list_gallery(folder):
    """Showcase entries: NAME.mp4 + NAME.json {title, prompt, mode, image, pipeline}, ordered by "order" then name."""
    if folder is None:
        return {"items": []}
    items = []
    for video in Path(folder).glob("*.mp4"):
        try:
            meta = json.loads(video.with_suffix(".json").read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(meta, dict) or not isinstance(meta.get("prompt"), str):
            continue
        image = meta.get("image")
        if image and not (Path(folder) / image).is_file():
            image = None
        items.append({"name": video.stem, "video": video.name, "image": image,
                      "title": meta.get("title") or video.stem.replace("-", " "),
                      "title_zh": meta.get("title_zh"), "prompt": meta["prompt"],
                      "mode": meta.get("mode", "first_frame" if image else "text"),
                      "pipeline": meta.get("pipeline", "one_stage"), "order": meta.get("order", 1000)})
    items.sort(key=lambda item: (item["order"], item["name"]))
    return {"items": items}


def gallery_file(folder, name):
    """A file directly inside the gallery folder (no subpaths), with its media type."""
    if folder is None or not name or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError("Gallery file unavailable")
    path = Path(folder) / name
    media = GALLERY_MEDIA.get(path.suffix.lower())
    if media is None or not path.is_file():
        raise ValueError("Gallery file unavailable")
    return path, media


def register():
    import asyncio
    from aiohttp import web
    import folder_paths
    from server import PromptServer
    from . import bridge, config
    from .comfy_nodes import PROGRESS
    server = PromptServer.instance
    if server is None or getattr(server, "_solh3_routes", False):
        return
    server._solh3_routes = True
    routes, no_store = server.routes, {"Cache-Control": "no-store"}

    def history(prompt_id):
        rows = server.prompt_queue.get_history(prompt_id=prompt_id).get(prompt_id, {})
        for output in rows.get("outputs", {}).values():
            for key in ("images", "videos", "gifs"):
                for item in output.get(key, []):
                    if item.get("type") == "output":
                        yield item.get("subfolder", ""), item["filename"]

    @routes.get("/solh3/capabilities")
    async def capabilities(request):
        return web.json_response(config.capabilities(), headers=no_store)

    @routes.get("/solh3/gallery")
    async def gallery(request):
        rows = await asyncio.to_thread(list_gallery, config.gallery_dir())
        return web.json_response(rows, headers=no_store)

    @routes.get("/solh3/gallery/file")
    async def gallery_media(request):
        try:
            path, media = gallery_file(config.gallery_dir(), request.query.get("name"))
        except ValueError:
            raise web.HTTPNotFound(text="Gallery file unavailable") from None
        # FileResponse serves byte ranges, so the browser can seek and loop the clips.
        return web.FileResponse(path, headers={"Content-Type": media, "Cache-Control": "private, max-age=3600"})

    @routes.get("/solh3/progress")
    async def progress(request):
        return web.json_response({"progress": list(PROGRESS.values())}, headers=no_store)

    @routes.post("/solh3/cancel")
    async def cancel(request):
        import comfy.model_management as mm
        mm.interrupt_current_processing()
        # The node's cancel poll also closes the worker; this covers a stuck poll.
        await asyncio.to_thread(bridge.close)
        return web.json_response({"status": "cancelling"})

    @routes.get("/solh3/library")
    async def library(request):
        try:
            limit = max(1, min(200, int(request.query.get("limit", "24"))))
        except ValueError:
            raise web.HTTPBadRequest(text="limit must be an integer") from None
        rows = await asyncio.to_thread(list_videos, folder_paths.get_output_directory(),
                                       limit, lambda pid: list(history(pid)))
        return web.json_response(rows, headers=no_store)

    @routes.get("/solh3/library/thumbnail")
    async def preview(request):
        try:
            data = await asyncio.to_thread(thumbnail, folder_paths.get_output_directory(),
                                           request.query.get("id"))
        except Exception:
            raise web.HTTPNotFound(text="Thumbnail unavailable") from None
        return web.Response(body=data, content_type="image/jpeg",
                            headers={"Cache-Control": "private, max-age=86400"})
