# Sol-H3 Studio v1 — frontend/harness spec

Goal (mentor brief): ComfyUI is the base; "the frontend acts as a harness" connected
to a locally deployed Sol engine, later also the B200 Sol-H3 API. Users should not need
the node graph. Pattern follows FreeVideo (github.com/FlashML-org/FreeVideo, web/studio.js):
a branded full-screen `<dialog>` over the ComfyUI canvas whose controls read/write the
widgets of one generate node; submit = `app.graphToPrompt()` + `api.queuePrompt()`.

## Scope v1 (agreed with user)
Prompt; mode Text / First frame / Reference image(s); aspect 16:9 / 9:16 / 1:1 (v1: 16:9
enabled, others shown disabled "coming soon" unless backend reports support); duration
5 s (v1 fixed, 10/15 s disabled); quality tier; seed (random / fixed); Generate with live
progress (stage + step k/8 + elapsed); result player with download; recent videos rail;
library view; backend switch Local RTX 5090 / Remote B200 API (remote = stub: listed,
disabled, "coming soon"). NOT in v1: launcher, agent, editor, chain/long video, LoRA UI.

## Backend contract the frontend relies on
Node `SolH3Generate` (keep existing id). Widgets (hidden in node view, edited by Studio):
`profile` (combo), `prompt` (STRING), `seed` (INT), `keep_loaded` (BOOL), plus new:
`mode` (combo: text|first_frame|reference), `image` (STRING, comma-separated uploaded
filenames in ComfyUI input dir; empty for text), `backend` (combo: local|remote).
Profiles come from `GET /solh3/capabilities` → `{profiles:[{name, label, steps, modes:[...],
default:bool}], backends:[{id:'local', label, available:true}, {id:'remote', label, available:false}]}`.

Routes (aiohttp on `PromptServer.instance.routes`, all prefixed `/solh3/`):
- `GET /solh3/capabilities` as above.
- `GET /solh3/progress` → `{progress:[{node, prompt_id, stage, step, steps, elapsed_s, message}]}` (polling fallback).
- Live progress: server `send_sync('solh3_progress', {...same fields})`.
- `POST /solh3/cancel` → interrupts the current request (calls ComfyUI interrupt + bridge close).
- `GET /solh3/library?limit=N` → `{items:[{id, file, subfolder, type, prompt, mode, profile, seed, e2e_s, created}]}` newest first, from outputs saved by SaveVideo under `SolH3/` with a sidecar JSON.
- `GET /solh3/library/thumbnail?id=` → JPEG poster (first frame), cached.
- Upload: reuse ComfyUI's existing `POST /upload/image` (no new route).

## Frontend files (web/)
- `solh3.js`: `app.registerExtension({name:'SolH3.Studio'})`; `beforeRegisterNodeDef` hides the new
  widgets and adds an "Open Sol-H3 Studio" button on SolH3Generate; `setup()` adds a small
  floating pill (wordmark + "Studio / Nodes" switch); `afterConfigureGraph` auto-opens Studio when
  `graph.extra.solh3_studio === true`.
- `studio.js` + `studio.css`: the dialog. Header: wordmark "Sol-H3" + subtitle "in ComfyUI",
  backend chip, Studio/Nodes switch. Body two columns: left controls (prompt textarea with
  character count and example-prompt chips; mode segmented control; image drop zone with
  thumbnails for first_frame/reference; aspect/duration chips; quality segmented; seed; big
  Generate button), right: stage (progress card with animated bar, stage name, step k/8, elapsed;
  then video player), below it recent-videos rail (last 8). Library opens as an overlay grid.
- `theme.css`: tokens scoped to `.sh-*` only; never restyle ComfyUI. Dark, premium, one accent
  (Sol green #76B900 family) + neutral graphite surfaces; Inter/system font; 8/12/16 radii.
- `wordmark.js` + `assets/solh3.svg`: simple typographic wordmark.
- Keep existing `timing.js` behaviour for node view.

## Template
`example_workflows/Sol-H3-Studio.json`: SolH3Generate → SaveVideo (`filename_prefix: SolH3/studio`),
`extra.solh3_studio: true`. Existing t2v templates remain for node users.

## Quality bar
Looks like a product, not a debug panel: generous spacing, real hierarchy, hover/focus
states, reduced-motion respected, works at 1280×800 and 1920×1080, zero console errors.
Bilingual strings via a tiny `t(en, zh)` helper keyed on `navigator.language`.
