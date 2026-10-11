# Racing render examples

The default example is **B10_e-track-3_bluehour**: an e-track-3 drive with
blue-hour lighting. Its prompt is also the default in
`../configs/assets.example.json`.

- `requests_A.jsonl`: 20 different scenes, including dirt/desert tracks,
  urban roads, forest, canyon, snow, and daylight/sunset/night appearances.
- `requests_B.jsonl`: 10 reference trajectories with four appearance
  variants each, for 40 examples.

Each record pairs a semantic reference video, a reference picture, and an
unaltered prompt. Settings are 1450 requested output frames, 24 fps, and
1344 × 768. These are request parameters, not measured source frame counts.

## Download the example media from GitHub

Videos and pictures are GitHub Release assets rather than Git history.
From `models/streamrender_h3`:

```bash
curl -fL --retry 3 -o streamrender-h3-examples.tar \
  https://github.com/NVlabs/Sana/releases/download/streamrender-h3-examples-v1/streamrender-h3-examples.tar
tar -xf streamrender-h3-examples.tar
```

Extraction adds `examples/media/` (30 semantic videos, 60 pictures) and
`examples/media_inventory.json` with file sizes and SHA-256 hashes. This
directory is Git-ignored. JSONL paths are relative to the model folder.

## Download weights and start a render

The public checkpoint bundle is [yitongl/StreamRender-H3](https://huggingface.co/yitongl/StreamRender-H3).
Follow the environment and checkpoint-download instructions in the
[runtime README](../README.md). After downloading the weights and running
the bundle's `download_assets.py`, use the resulting `assets.json`:

```bash
python scripts/prepare_example.py --list
python scripts/prepare_example.py --assets /absolute/path/to/StreamRender-H3/assets.json
GPUS=8 PYTHON_BIN=python bash scripts/launch_local.sh --assets configs/example.local.json
```

The selection script defaults to B10_e-track-3_bluehour. It preserves weight
paths, installs the selected prompt and picture, and refuses to overwrite
an existing local manifest. To select another request:

```bash
python scripts/prepare_example.py \
  --assets /absolute/path/to/StreamRender-H3/assets.json \
  --example A11_g-track-2_snow --output configs/snow.local.json
```

Launch the worker with that manifest before replaying the corresponding
video. The worker's prompt and reference picture are fixed at startup;
changing only the replay name does not change model conditioning.

After the worker reports readiness, replay the selected semantic video
from a second terminal (Python environment needs PyAV, Pillow and requests):

```bash
python scripts/replay_example.py --example B10_e-track-3_bluehour
```

Feedback JPEG frames are written to `outputs/replay/`. `--max-frames 107`
limits a run, and `--server http://HOST:8765` selects a tunneled worker.
The client preserves frame order and retries the same frame on backpressure.
It closes the session after draining complete chunks; a partial tail is
discarded. This client is for example playback, not latency benchmarking.

For interactive driving, open the included Three.js browser UI instead.
These prerecorded examples cover more tracks than the single included
live-engine scene; replay uses each example's actual semantic trajectory.
There are no player-action transcripts in these two request lists.

The example contributor approved publication. The model's upstream licenses
remain applicable; publication of the examples does not relicense
third-party game assets.
