# Contributing

Thanks for considering a contribution.

Symposium is intentionally small. Contributions should make the loop clearer, safer, or easier to install without turning the project into a large framework.

## Development Principles

- Keep skills plain Markdown.
- Keep the canonical source in `skills/`.
- Do not maintain separate Codex and Claude source files.
- Prefer runtime-neutral wording.
- Keep scratch state under `.symposium/scratch`.
- Do not let the model auto-adopt user decisions.

## Skill Quality Checklist

Every skill should include:

- `SKILL.md`
- `plan.md`
- `test_cases.json`
- Clear triggers
- Clear "Do NOT use when" boundaries
- A `vs ...` distinction from nearby skills

## Test Locally

Validate JSON:

```bash
for f in skills/*/test_cases.json; do jq empty "$f" || exit 1; done
```

Install into a local Codex copy:

```bash
./scripts/install.sh --target codex --scope project
```

Install into a local Claude copy:

```bash
./scripts/install.sh --target claude --scope project
```

## Pull Request Guidance

Good PRs are small and focused:

- One skill behavior change, or
- One documentation improvement, or
- One installer improvement.

Please include:

- What changed
- Why it matters
- How you tested it

## Ideas Worth Contributing

- More realistic `test_cases.json` examples
- Example Seeds from real workflows
- Better installer support for other agent runtimes
- A small demo recording or terminal transcript
- Translations that preserve the same canonical skill behavior
