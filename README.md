# SANA project pages

This branch contains only the static site code for the SANA project pages.

All page media and downloadable data live in one Hugging Face dataset:
[`Efficient-Large-Model/Sana-assets`](https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets).
Do not commit videos, images, audio, PDFs, HLS segments, or generated datasets to
this branch. Upload them under a page-specific directory in `Sana-assets` and
reference the corresponding `resolve/main/...` URL from the site code.

The SANA-Video 2.0 curated samples from 2026-08-12 are stored together under
`Video2/assets/curated-20260812/`, including the 20 videos, 10 conditioning
frames, posters, prompts, selection summary, and complete metadata.
