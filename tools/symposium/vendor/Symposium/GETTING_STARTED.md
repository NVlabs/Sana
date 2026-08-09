# Getting Started

This guide gets Symposium running in a local project and walks through the first interview.

## 1. Install The Skills

From the Symposium repo:

```bash
./scripts/install.sh --target codex --scope project
```

For Claude:

```bash
./scripts/install.sh --target claude --scope project
```

The installer copies `skills/` into `.codex/skills` or `.claude/skills`.

## 2. Start With A Vague Request

Use a deliberately fuzzy task:

```text
Use interview-harness on: I want to build a simple task management CLI.
```

The harness will:

1. Create a Seed if one does not exist.
2. Ask about hidden meanings and scope.
3. Review blind spots.
4. Add an ontology boundary.
5. Stop when meaning converges or the safety valve triggers.

## 3. Answer Conservatively

Symposium works best when you answer with concrete boundaries:

```text
This is only for personal local use.
No cloud sync.
No team collaboration.
It should store tasks in a local JSON file.
```

The agent should not invent these details for you.

## 4. Inspect The Scratch Files

After a run, check:

```bash
ls .symposium/scratch
```

Important files:

- `socrates.md`: the interview cycles plus latest Seed and Ontology.
- `evolve-step.md`: blind-spot transfeat and accepted changes.
- `interview-harness.md`: ontology snapshots and convergence scores.

## 5. Use The Final Seed

Once the harness returns a final Seed, give it to your coding agent:

```text
Implement this Seed. Treat the acceptance_criteria as the completion gate.
```

You can also paste the Seed into an issue, PRD, or task tracker.

## Common Commands

Install locally for Codex:

```bash
./scripts/install.sh --target codex --scope project
```

Install locally for Claude:

```bash
./scripts/install.sh --target claude --scope project
```

Install globally for Codex:

```bash
./scripts/install.sh --target codex --scope global
```

Install globally for Claude:

```bash
./scripts/install.sh --target claude --scope global
```

## Troubleshooting

### The agent does not see the skills

Restart the agent session after installing. Some runtimes load skills only at session start.

### The agent writes to `.claude/scratch` or `.codex/scratch`

The canonical Symposium skills use `.symposium/scratch`. If your agent writes elsewhere, it is likely using an older copied skill. Run the installer again.

### The loop asks too many questions

Use a smaller goal. Symposium is designed for compact Seeds, not full product strategy documents.

### The result is too broad

Answer the boundary questions with exclusions. The best Seeds say what is out of scope.
