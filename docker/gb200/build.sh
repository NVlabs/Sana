#!/usr/bin/env bash
# Build the image with a tag that names the exact git revision: <version>-g<short sha>[-dirty], plus
# <version>. Needs an arm64 Docker host for GB200 (Apple-silicon Mac + colima works), or buildx elsewhere.
set -euo pipefail
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || (cd "$(dirname "$0")/../.." && pwd))
cd "$ROOT"
VERSION=${VERSION:-0.6.0}
SHA=${GIT_SHA:-$(git rev-parse HEAD)}
TAG="${VERSION}-g${SHA:0:9}"
if git rev-parse --git-dir >/dev/null 2>&1 && ! git diff --quiet HEAD; then TAG="${TAG}-dirty"; fi
IMAGE=${IMAGE:-sol-engine-gb200}
PLATFORM=${PLATFORM:-linux/arm64}
docker buildx build --platform "$PLATFORM" --load -f docker/gb200/Dockerfile \
  --build-arg GIT_SHA="$SHA" --build-arg VERSION="$TAG" -t "$IMAGE:$TAG" -t "$IMAGE:$VERSION" .
echo "built $IMAGE:$TAG"
