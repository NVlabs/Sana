"""Run the same bridge as the ComfyUI node, e.g. for a measured demo."""
import argparse
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from sol_h3_engine.config import load_profile
from sol_h3_engine import bridge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    try:
        for _ in range(args.repeat):
            result = bridge.generate(load_profile(args.profile), args.output, args.prompt, args.seed)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
