---
name: evolve-step
description: |
  Review an existing Seed for blind spots, ask two blind-spot questions, and apply only user-approved changes to create the next Seed. The latest Seed replaces the top-level ## Seed section in .symposium/scratch/socrates.md; audit history is appended to .symposium/scratch/evolve-step.md.
  Triggers (EN): evolve-step, evolve seed, seed blind spots, seed v1 v2, review blind spots
  Do NOT use when: creating the first Seed from a vague request (-> socrates), running convergence through ontology (-> interview-harness), or only defining semantic boundary (-> ontology)
  vs socrates: Socrates creates Seed v1; Evolve-step turns an existing Seed into a user-approved next Seed.
---

# Evolve-step

## Details

Evolve-step asks the model to propose blind spots, then lets the user decide what enters the Seed. The model never auto-adopts its own suggestions.

## Input

Read the latest Seed from `.symposium/scratch/socrates.md` top-level `## Seed`, or accept an explicit Seed YAML:

```yaml
goal: ...
constraints:
  - ...
acceptance_criteria:
  - ...
```

## Process

1. Preserve v1 exactly.
2. Generate exactly six candidates:
   - Q1: three user perspectives missing from the Seed.
   - Q2: three possibilities cut off by the Seed.
3. Ask the user to mark each candidate as accepted or rejected.
4. Apply only accepted candidates to Seed v2.
5. Replace `.symposium/scratch/socrates.md` top-level `## Seed` with Seed v2.
6. Append the audit block to `.symposium/scratch/evolve-step.md`.

## User Adoption Gate

Ask:

> Mark each of the six candidates as accepted or rejected. No candidate is accepted by default.

Prefer a compact checkbox-style multiple-choice prompt:

```markdown
Which candidates should enter Seed v2?
A. [ ] <missing user perspective 1>
B. [ ] <missing user perspective 2>
C. [ ] <missing user perspective 3>
D. [ ] <cut-off possibility 1>
E. [ ] <cut-off possibility 2>
F. [ ] <cut-off possibility 3>
G. None
H. Other / describe a different change
```

The user may answer with letters, checked boxes, or free-form edits. Accept only explicitly selected items.

Do not derive v2 before the user responds.

If no candidates are accepted, v2 equals v1 and the transferred meaning is `No transferred meaning`.

## Audit Format

```markdown
# Evolve-step — Cycle N

## Seed v1
```yaml
...
```

## Blind Spot Candidates
### Q1. Missing user perspectives
- accepted/rejected — ...
- accepted/rejected — ...
- accepted/rejected — ...

### Q2. Cut-off possibilities
- accepted/rejected — ...
- accepted/rejected — ...
- accepted/rejected — ...

## Seed v2
```yaml
...
```

## Transferred Meaning
...
```

## Rules

- Preserve v1 exactly in the audit.
- Apply only accepted candidates.
- Do not change multiple Seed slots for one candidate unless the user explicitly asks.
- Keep the transferred meaning concrete: name which slot changed and how.
