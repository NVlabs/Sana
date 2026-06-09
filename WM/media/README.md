# public/media/

This directory is **populated at build time** and is git-ignored on `main`.

- Locally, run `./scripts/sync-media.sh` from the repo root to mirror the
  contents of the `media` branch (Git LFS) into `videos/`, `overlays/`, and
  `figures/`.
- In CI, the GitHub Pages workflow checks the `media` branch out with
  `lfs: true` and copies the same layout in before `vite build`.

To add or replace media, push to the `media` branch — see the project
`README.md` for the workflow.
