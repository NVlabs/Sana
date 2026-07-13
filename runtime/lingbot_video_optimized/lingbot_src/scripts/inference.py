from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lingbot_video.inference_backend import default_backend_argv  # noqa: E402
from lingbot_video.runner import main as runner_main  # noqa: E402


def main() -> None:
    sys.argv[1:] = default_backend_argv(sys.argv[1:])
    runner_main()


if __name__ == "__main__":
    main()
