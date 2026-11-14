# Sol-Engine images for InferenceX (MiniMax-H3 video)

Two images serve MiniMax-H3 for the InferenceX VideoGenX board (`h3-v2.yaml` release-v1):

| Platform | GPUs | InferenceX PR | Source commit | Image |
|---|---|---|---|---|
| GB200 (`linux/arm64`) | 1, 2, 4 | #4940 | [`d23f2ebf5`](https://github.com/NVlabs/Sana/commit/d23f2ebf58b69292ffb6a0ab850a989cf0685323) | `ghcr.io/lyttttt3333/sol-engine-gb200:0.6.0-gd23f2ebf5` |
| H100 (`linux/amd64`) | 2, 4, 8 | #4938 | [`d8fee0c4f`](https://github.com/NVlabs/Sana/commit/d8fee0c4f636a22f5a1bd96837c26f2084428341) | `ghcr.io/lyttttt3333/sol-engine-h100:0.6.0-gd8fee0c4f` |

Each image is built from this repository at its source commit, with `docker/gb200/Dockerfile`
(+ `install.sh`, `requirements.txt`) at that commit. Both commits are parents of the merge on
branch `inferencex/h3`, which also carries this file and the package lists.

Attention is exact dense attention and every tensor is BF16; the InferenceX recipes pass
`--attention dense --precision bf16`.

Both images use the base image `nvidia/cuda:13.0.2-runtime-ubuntu24.04`, index
`sha256:6a0e31b59e70890446f2c17356d6efc0d54260090a8c63d6ca7c2ad049db95d2`
(linux/arm64 `sha256:bbfc230dfd4c3bb2eadacdc22cb7e2ea3453e93bb3ee40e241357eb0bcd5c6a6`,
linux/amd64 `sha256:121b25126d81cddeae4c048aa46e94050514ca6b8609a124027b38073e7ef4b6`).

## GB200

| | |
|---|---|
| Source | `NVlabs/Sana` commit [`d23f2ebf58b69292ffb6a0ab850a989cf0685323`](https://github.com/NVlabs/Sana/commit/d23f2ebf58b69292ffb6a0ab850a989cf0685323) |
| Platform | `linux/arm64` |
| Image | `ghcr.io/lyttttt3333/sol-engine-gb200:0.6.0-gd23f2ebf5@sha256:6eebd484e018041f74af84b088da5bc4772f19feeab7a0d2e41613cb5011bc9d` |
| Python packages | [`inferencex-gb200-packages.txt`](./inferencex-gb200-packages.txt): every package installed in `/opt/sol-engine/venv`, with versions |

```bash
git clone https://github.com/NVlabs/Sana && cd Sana
git checkout d23f2ebf58b69292ffb6a0ab850a989cf0685323
docker buildx build --platform linux/arm64 \
  -f docker/gb200/Dockerfile \
  --build-arg BASE=nvidia/cuda:13.0.2-runtime-ubuntu24.04@sha256:6a0e31b59e70890446f2c17356d6efc0d54260090a8c63d6ca7c2ad049db95d2 \
  --build-arg GIT_SHA=d23f2ebf58b69292ffb6a0ab850a989cf0685323 \
  --build-arg VERSION=0.6.0-gd23f2ebf5 \
  -t sol-engine-gb200:0.6.0-gd23f2ebf5 --load .
```

Measured with the published image, run locally on our own GB200 nodes:

| GPUs | client p50 | valid |
|---|---|---|
| 1 | 119.55 s | 20/20 |
| 2 | 66.54 s | 20/20 |
| 4 | 34.43 s | 20/20 |

## H100

| | |
|---|---|
| Source | `NVlabs/Sana` commit [`d8fee0c4f636a22f5a1bd96837c26f2084428341`](https://github.com/NVlabs/Sana/commit/d8fee0c4f636a22f5a1bd96837c26f2084428341) |
| Platform | `linux/amd64` |
| Image | `ghcr.io/lyttttt3333/sol-engine-h100:0.6.0-gd8fee0c4f@sha256:c6d213b473e89634157e67352fd0ebcc9dd59b9d03b56450c3d0665365e735ba` |
| Python packages | [`inferencex-h100-packages.txt`](./inferencex-h100-packages.txt): every package installed in `/opt/sol-engine/venv`, with versions |

```bash
git clone https://github.com/NVlabs/Sana && cd Sana
git checkout d8fee0c4f636a22f5a1bd96837c26f2084428341
docker buildx build --platform linux/amd64 \
  -f docker/gb200/Dockerfile \
  --build-arg BASE=nvidia/cuda:13.0.2-runtime-ubuntu24.04@sha256:6a0e31b59e70890446f2c17356d6efc0d54260090a8c63d6ca7c2ad049db95d2 \
  --build-arg GIT_SHA=d8fee0c4f636a22f5a1bd96837c26f2084428341 \
  --build-arg VERSION=0.6.0-gd8fee0c4f \
  -t sol-engine-h100:0.6.0-gd8fee0c4f --load .
```

Measured with the published image, run locally on our own H100 nodes:

| GPUs | client p50 | valid |
|---|---|---|
| 2 | 143.92 s | 20/20 |
| 4 | 74.46 s | 20/20 |
| 8 | 39.39 s | 20/20 |

## How the published images were built

With kaniko (no Docker daemon on our clusters), from a checkout of the source commit:
`/kaniko/executor --context dir://<checkout> --dockerfile docker/gb200/Dockerfile --custom-platform <platform>
--build-arg GIT_SHA=<commit> --build-arg VERSION=0.6.0-g<commit[:9]> --build-arg APT_NO_SANDBOX=1 --snapshot-mode redo`.

## Checking a rebuild

A rebuild will not have the same image digest (layer timestamps, and Ubuntu packages from the
base image's apt mirrors change over time). Check it against the published image by its Python
packages instead (from the same clone; shown for H100, use `gb200` for GB200):

```bash
docker run --rm sol-engine-h100:0.6.0-gd8fee0c4f /opt/sol-engine/venv/bin/python -c '
import importlib.metadata as m, re
rows = {(re.sub(r"[-_.]+", "_", d.metadata["Name"]).lower(), d.version) for d in m.distributions()}
print("\n".join(f"{n}=={v}" for n, v in sorted(rows)))' \
  | diff - <(git show origin/inferencex/h3:docker/gb200/inferencex-h100-packages.txt)
```

and by running the InferenceX recipe `h3/sol-engine/{h100,gb200}-bf16/serving-client.yaml` with it.

The measurements above use InferenceX `infx.bench video serve` / `video client` on h3-v2
release-v1 (20 prompts, C1, 49 denoising updates).
