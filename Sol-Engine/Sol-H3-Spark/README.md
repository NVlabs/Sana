# Sol-H3 on one NVIDIA DGX Spark

Static project page, separate from the existing eight-B300 Sol-H3 release.
Serve this directory over HTTP; no Node server or build step is required.
The page retains the blog's dark/light themes, responsive navigation,
hero loop, synchronized showcase playback, prompt details and BibTeX copy.
The hero defaults to sound on; showcase videos default to muted. Browsers
that block audible autoplay retain the hero's click-to-play control.

## Assets

All video, poster, logo and favicon assets are hosted in
[Sana-assets](https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets/tree/main/Sol-Engine/Sol-H3-Spark/20260907).
Existing asset URLs retain their pinned dataset revisions. The four new
showcase examples use revision `efed7c62d78352b703adaf1d5a1d0a8de445c8fd`
under `Sol-Engine/Sol-H3-Spark/20260910/showcase/`.
No access token or authentication is needed to view them.

The homepage hero is the 35.333-second animated Sol-H3 on Spark introduction,
at 1344x768 / 24 FPS with native dialogue and an instrumental music bed. Its
MP4 and frame-derived poster are pinned to Sana-assets revision
`e237daa52e404622250583edcff46d027d02241c` under
`Sol-Engine/Sol-H3-Spark/20260910/hero/`. The film was produced with Base H3
Ref2VA (50 scheduler points / 49 DiT evaluations, no LoRA), not with
the 56-second Spark inference pipeline. Hero labels identify it as an
introduction rather than a measured Spark generation. The benchmark claims
and existing showcase media remain unchanged; older showreel assets are retained.

## Content and maintenance

`index.html` retains the current blog's rendered content; `style.css` is its
compiled stylesheet, based on the organization-owned Sol-H3 design, with
Spark-specific layout changes. `light.css` provides the optional light theme.
`script.js` implements the standalone playback, navigation and copy controls.
Keep relative code links and HF-hosted media URLs so the page works under
`/Sana/Sol-Engine/Sol-H3-Spark/` without a server runtime or root-path rewrite.

The headline is the recorded mean of three warm resident requests (56.17 s).
The FastH3 comparison cites its published 374 s cold/phased-loading run;
the visible methodology note explicitly identifies the different timing
regimes and geometry. It is not a matched benchmark or quality-equivalence
claim. These values and the frozen three-step pipeline are unchanged here.

## Showcase additions

The first four examples are Snow Leopard Ridge, Wildflower Firefly Portrait,
Fountain Pen Blue Ink and Astronaut Reentry Cockpit, in that order. The eight
previous examples follow in their existing order. Pagination remains two
examples per page, with six pages and muted playback.

The videos use the existing two-stage pipeline. They and their posters are
hosted on HF under descriptive public paths, with no media binaries committed
to this branch. The current homepage introduction and timing claims are retained.
