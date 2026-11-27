# Video media workflow

This directory intentionally contains no media binaries. Every SANA-Video 2.5
clip and poster lives in the shared
[`Efficient-Large-Model/Sana-assets`](https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets)
dataset under `Video25/media/`, and `../data.js` references the corresponding
`resolve/main/...` URLs.

- `media/src/{talk,t2av,long}/*.mp4`: the sampler's original H.264 + AAC output,
  remuxed losslessly for streaming (never re-encoded).
- `media/{talk,t2av,long}/*.jpg`: posters shown until playback begins.
