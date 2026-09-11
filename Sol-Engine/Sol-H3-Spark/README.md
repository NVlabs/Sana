# Sol-H3 on one NVIDIA DGX Spark

Static research project page, separate from the eight-B300 Sol-H3 release.
Serve this directory over HTTP; no build step or application server is required.

## Reviewed release

`index.html` is the self-contained HTML/CSS/JavaScript artifact reviewed at
[the HF preview](https://hp-l33-minimax-h3-dgx-spark.static.hf.space/sol-h3-lieflat/index.html),
Space revision `2ad5a0c8004f69435e092f2c24cb88ca3c9f626c`. It retains light/dark
themes, responsive navigation, two-at-a-time showcase playback and pagination,
exact prompts, animated pipeline previews, timing details and BibTeX copy.
The older `style.css`, `light.css` and `script.js` remain for historical reference;
this release does not load them.

Both Code buttons link to the
[Spark-specific pipeline](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/Sol-H3-Spark).
The page reports the recorded 56.17-second resident mean as 56 seconds and
includes a single percentage-stacked component timing breakdown. Latency values
sit at the foreground bar ends; the 6.7× annotation links the external
374-second 4-step LoRA reference to Sol-H3, not to the quantized row. It is not
a matched cold/hot comparison. The Sol-Attn card's 3.1× figure is a shape-matched
GB10 kernel benchmark against FA4, not an end-to-end speedup.

## Assets and showcase

Videos, posters and logos remain hosted on
[Sana-assets](https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets).
Their URLs are pinned to dataset revisions; viewing needs no token. No media
binaries or credentials are included in this directory.

The homepage uses the supplied `sol-h3-on-spark-three-act-cow-v1.mp4`
without transcoding: 44.958 seconds, 1344 × 768, 24 FPS, with audio. The file
and its frame-derived poster are pinned to dataset revision
`6b114b40fe4851ba5a012559aa2c8b9c610c373c`, under
`Sol-Engine/Sol-H3-Spark/20260911/hero/`. The introduction is not a timed
56-second Spark generation. Audible autoplay remains browser-dependent;
the hero has a click-to-play control, and showcase playback starts muted.

The 12 showcase clips are interleaved across six pages:

1. Strawberry heist / Mandarin radio reunion
2. Snow leopard ridge / Wildflower firefly portrait
3. English seaside conversation / Sky harbor cloud animation
4. Fountain pen blue ink / Astronaut reentry cockpit
5. Crystal ball / Film projector
6. Night-train chess / Orbital greenhouse

The newer selected stories and posters use dataset revision
`b122ef1a16cc33f8cc32bc17f40d85326ceadaf8` under
`Sol-Engine/Sol-H3-Spark/20260910/selected-stories/`. Retained examples keep
their existing pinned URLs. Previous public assets have not been deleted.

## Design notices

This version includes Lieflat-inspired presentation from the reviewed preview.
Preserve `LIEFLAT-LICENSE.txt` and `THIRD-PARTY-NOTICES.txt` alongside the HTML.
The Lieflat-derived material retains its separate PolyForm Noncommercial 1.0.0
terms; inclusion here does not relicense it under the repository's license.
Inter is loaded from Google Fonts under the SIL Open Font License 1.1.
