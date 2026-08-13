# Video media workflow

This directory intentionally contains no media binaries. Upload every Video 2.0
asset to the shared
[`Efficient-Large-Model/Sana-assets`](https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets)
dataset under `Video2/`, then use a path relative to the `assetBase` declared in
`../media-config.js`.

Recommended per video:

- `poster`: WebP, AVIF, or JPEG.
- `hls`: a multibitrate HLS playlist when available.
- `mp4`: a fast-start MP4 fallback (`ffmpeg -movflags +faststart`).

The page loads media only as it becomes visible. Data Saver and reduced-motion
users remain on posters unless they explicitly open a video.
