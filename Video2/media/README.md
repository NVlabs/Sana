# Video media workflow

The site is poster-first. Add final URLs to the existing `intro` and `sections`
entries in `../media-config.js`; the HTML does not need to change.

Recommended per video:

- `poster`: WebP or AVIF, roughly 20–100 KB.
- `hls`: a multibitrate HLS `master.m3u8` (preferred).
- `mp4`: a fast-start MP4 fallback (`ffmpeg -movflags +faststart`).

Each section may contain any number of videos. Its two-row viewport scrolls
continuously with wheel or trackpad distance rather than snapping by row. Rows
softly fade into the background at the top and bottom edges; the upper fade
extends behind the section title while the two focused rows remain fully opaque.
Wheel input is captured at the demo-stage
boundary so scrolling feels identical whether the pointer is over the grid or
the surrounding background. At the first and final demo boundaries, wheel
distance is explicitly handed back to the document so the title and citation
pages remain reachable. After the final row is reached, the next gesture changes
section; trackpad inertia is filtered at that boundary.

Large wheel deltas are clamped before they cross the demo entry anchors. A
section transition requires an actual idle gap between gestures, preventing a
single high-speed trackpad flick from being mistaken for several section jumps.

The title page and citation page remain normal document sections. Scrolling is
captured only after the demo stage completely fills the viewport and is released
again after its final row. The title-page media wall loops horizontally; until
video sources are present, the same motion uses the lightweight posters. Hero
cards are purely decorative: pointer interaction never zooms them, pauses the
wall, or opens the fullscreen player.

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
