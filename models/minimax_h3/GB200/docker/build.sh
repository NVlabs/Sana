#!/usr/bin/env bash
# Build the image with a tag that names the exact git revision:
#   <version>-g<short sha>[-dirty], plus the plain <version>. Run on an arm64 Docker host for GB200
#   (or pass PLATFORM=linux/arm64 to buildx on another host).
set -euo pipefail
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$ROOT"
VERSION=${VERSION:-0.5.0}
SHA=$(git rev-parse HEAD)
TAG="${VERSION}-g${SHA:0:9}"
git diff --quiet HEAD || TAG="${TAG}-dirty"
IMAGE=${IMAGE:-sol-engine-h3-gb200}
PLATFORM=${PLATFORM:-linux/arm64}
docker buildx build --platform "$PLATFORM" --load \
  -f models/minimax_h3/GB200/docker/Dockerfile \
  --build-arg GIT_SHA="$SHA" --build-arg VERSION="$TAG" \
  -t "$IMAGE:$TAG" -t "$IMAGE:$VERSION" .
echo "built $IMAGE:$TAG"
