# Video media workflow

The site is poster-first. Add final URLs to the existing `intro` and `sections`
entries in `../media-config.js`; the HTML does not need to change.

Recommended per video:

- `poster`: WebP or AVIF, roughly 20–100 KB.
- `hls`: a multibitrate HLS `master.m3u8` (preferred).
- `mp4`: a fast-start MP4 fallback (`ffmpeg -movflags +faststart`).

Each section may contain any number of videos. Its two-row viewport moves
continuously rather than snapping by row. Rows softly fade into the background
at the top and bottom edges; the upper fade extends behind the section title
while the two focused rows remain fully opaque.

The entire experience uses one native document scroll. Wheel, trackpad, touch,
keyboard, and scrollbar dragging all follow the same continuous timeline; no
wheel events are captured and no scroll position is forced at section
boundaries. The demo grid advances only after its sticky stage completely fills
the viewport. Each section keeps a symmetric visual scroll margin at both ends.
The margin continues moving the grid rather than freezing it: on entry, the
first row rises from the second-row position; on exit, the final visible row
continues from the second-row position into the first before the section may
change. Crossing the boundary then triggers a fixed-time animation for the
title, description, and grid. That animation is not scrubbed by scroll, so the
scrollbar cannot rest on a mixed half-old, half-new section. The same path plays
in reverse when scrolling upward, and the last demo releases naturally into the
citation page.

The title and citation remain normal document sections. The title-page media
wall loops horizontally; until video sources are present, the same motion uses
the lightweight posters. Hero cards are purely decorative: pointer interaction
never zooms them, pauses the wall, or opens the fullscreen player.

Headline metrics use a compact overlay rather than a full-width panel, keeping
the moving media wall visually dominant.

Desktop demo grids use three columns and expose six videos in the focused
two-row band. Row height is derived from card width and capped near 16:9 instead
of stretching to fill available height. The hero wall likewise uses six media
tiles per set; compact screens fall back to two columns.

Streams load only on hover/focus or after opening the fullscreen player, so
quickly scrolling through the showreel does not fetch every video. Posters stay
visible until playback actually begins. Data Saver and reduced-motion users
stay on posters unless they explicitly open a video.
