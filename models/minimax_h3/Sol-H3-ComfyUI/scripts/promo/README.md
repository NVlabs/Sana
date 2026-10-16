# The "Sol-H3 in ComfyUI" film

`make_promo.py` cuts the film from generated clips and motion-graphics overlays rendered in a
headless browser. Every number on screen comes from `timing.json`.

| Part | Source |
|---|---|
| Hero, captions, latency chart, end card | `motion/scenes.html` (rendered by `motion/render.mjs`) |
| Studio walkthrough | `motion/studio_film.html`: the real `web/studio.js` with a scripted camera and cursor (rendered by `motion/render_film.mjs`) |
| Clips | first-frame showcase clips (`showcase/`), the Studio gallery |
| Music | `music.py` (synthesized) |

Requirements: Node.js with Playwright, Google Chrome, FFmpeg (`PROMO_FFMPEG`), Python 3 with
NumPy. Fetch the media first:

```bash
python3 scripts/fetch_assets.py promo gallery
python3 -m http.server 8777 &          # from this directory's root; the film loads ES modules
export PLAYWRIGHT_PATH=/path/to/node_modules/playwright PROMO_FFMPEG=$(which ffmpeg)
FILM_QUERY="one=67.2&two=35.5" node scripts/promo/motion/render_film.mjs b assets/studio-b.mp4
python3 scripts/promo/make_promo.py --clips assets/promo/clips --studio-films assets/studio-b.mp4 \
  --timing scripts/promo/timing.json --out sol-h3-in-comfyui.mp4
```

`FILM_QUERY` passes the measured one-stage and two-stage latencies shown on the Studio's
pipeline cards.

## Showcase clips

The first-frame clips were made in three steps; they are promotional content, not timing claims:

1. `showcase/qwen_keyframes.py keyframes.json OUT`: first frames with Qwen-Image-2.1.
2. `showcase/official_conditioner.py MODEL_DIR jobs.json OUT`: the official BF16 MiniMax-H3
   conditioner (needs a large-memory GPU).
3. `showcase/hyperflow_showcase.py OPTIONS.json PAYLOAD_DIR OUT`: the one-stage engine on an RTX 5090.
