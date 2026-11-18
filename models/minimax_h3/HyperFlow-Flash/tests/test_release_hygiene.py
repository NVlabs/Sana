"""The published tree carries no machine-specific paths and no scheduler job references."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [
    re.compile(r"/(?:home|Users|lustre|scratch|mnt|gpfs)/[A-Za-z0-9_.-]+"),
    re.compile(r"\binternal\b", re.IGNORECASE),
    re.compile(r"\b(?:sbatch|squeue|srun)\b"),
    re.compile(r"\bjob(?:s)? \d{5,}"),
]


def test_no_machine_specific_text():
    hits = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".md", ".json", ".txt", ".toml"} or path.name == Path(__file__).name:
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            hits += [f"{path.relative_to(ROOT)}:{n}: {line.strip()}" for p in PATTERNS if p.search(line)]
    assert not hits, "\n".join(hits)
