# SANA-Streaming Project Page Prototype

Static project-page prototype for SANA-Streaming.

Run locally:

```bash
cd /Users/yyzhao/Desktop/Paper/NeurIPS-2026/sana-streaming-page
python3 -m http.server 8088
```

Open:

```text
http://localhost:8088/
```

The page serves web-optimized local videos from:

```text
assets/media/
```

Regenerate those files from the original examples with:

```bash
python3 scripts/prepare_web_videos.py
```

If the source examples are not at `/Users/yyzhao/Downloads/sana-streaming-example`, set:

```bash
SANA_STREAMING_SOURCE_ROOT=/path/to/sana-streaming-example python3 scripts/prepare_web_videos.py
```
