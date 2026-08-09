"""Nothing the repository declares private may be tracked.

This is a public repository. Two things must stay out of it: Slurm job scripts,
which carry an account, a partition and a QoS that mean nothing off this
cluster, and the MiniMax-H3 GB200 SGLang runtime, which is kept local for
framework A/B work. `.gitignore` says both, and `.gitignore` was not enough:

  * an ignore rule does not untrack a file that was tracked before the rule
    existed, so two `.sbatch` files sat in the published tree while `*.sbatch`
    was in `.gitignore` the whole time; and
  * the SGLang rule was written `models/minimax_h3/gb200_sglang/`, and when the
    hardware directories were capitalised to match the vendor's spelling the
    rule stopped matching a directory that had merely changed case. Thirteen
    files were published by a rename that touched none of them.

Both failures are silent by construction -- a stale ignore rule looks exactly
like a satisfied one, and `git status` is clean either way. So assert the
intent directly against the index, which is the thing that actually gets
pushed, rather than trusting the rules to still describe it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> list[str]:
    out = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [line for line in out.splitlines() if line.strip()]


def test_no_tracked_file_is_gitignored() -> None:
    """The general rule, and the one that catches the next stale pattern.

    Any file that is both tracked and claimed by an ignore rule is a
    contradiction: someone wrote the rule meaning to keep it out. Whichever
    half is wrong, a human should decide -- silently publishing it is not the
    answer.
    """
    leaked = _git("ls-files", "-i", "-c", "--exclude-standard")
    assert not leaked, (
        "tracked files are claimed by a .gitignore rule, so they are published "
        "against the repository's own stated intent:\n  "
        + "\n  ".join(leaked)
        + "\nUntrack them with `git rm --cached <path>` (this keeps your local "
        "copy), or drop the ignore rule if it is the part that is wrong."
    )


def test_no_slurm_job_scripts_tracked() -> None:
    """Named separately from the rule above so the failure says what leaked.

    `*.sbatch` is the concentrated case: every one of these files carries
    `--account`, and most carry an absolute path through a personal home or
    Lustre directory. `docs/simple-launch.md` documents the four-line wrapper
    instead, with the account as a placeholder.
    """
    tracked = [f for f in _git("ls-files") if f.endswith(".sbatch")]
    assert not tracked, (
        "Slurm job scripts are tracked; they carry a cluster account:\n  "
        + "\n  ".join(tracked)
    )


def test_sglang_gb200_runtime_not_tracked() -> None:
    """GB200 publishes its Diffusers runtime. The SGLang one stays local."""
    tracked = [
        f for f in _git("ls-files") if f.startswith("models/minimax_h3/GB200_sglang")
    ]
    assert not tracked, (
        "the GB200 SGLang runtime is tracked; only the Diffusers runtime is "
        "published:\n  " + "\n  ".join(tracked)
    )


def test_ignore_rules_still_match_something() -> None:
    """A path rule that matches nothing on disk is how the last leak happened.

    Only literal directory rules are checked -- a glob like `*.mp4` legitimately
    matches nothing in a clean tree, and a rule for a runtime artefact
    directory legitimately does not exist yet. A rule naming a concrete
    directory that is absent, while a sibling differing only in case is
    present, is the exact stale-after-rename signature and nothing else.
    """
    stale = []
    for raw in (ROOT / ".gitignore").read_text().splitlines():
        rule = raw.strip()
        if not rule or rule.startswith(("#", "!")) or "*" in rule:
            continue
        target = ROOT / rule.rstrip("/")
        if target.exists() or not target.parent.is_dir():
            continue
        for sibling in target.parent.iterdir():
            if sibling.name.lower() == target.name.lower():
                stale.append(f"{rule}  ->  the directory is really {sibling.name}")
                break
    assert not stale, (
        "ignore rules whose target differs only in case from a directory that "
        "does exist -- these match nothing and quietly publish it:\n  "
        + "\n  ".join(stale)
    )
