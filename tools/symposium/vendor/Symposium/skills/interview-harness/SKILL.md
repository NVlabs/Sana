---
name: interview-harness
description: |
  Run Socrates, Evolve-step, and Ontology as a semi-automatic convergence loop. If a complete Seed already exists, skip Socrates. Iterate evolve-step and ontology until ontology similarity reaches 0.85 or the safety valve stops the loop.
  Triggers (EN): interview-harness, run interview harness, converge seed, socratic harness, close this into a seed
  Do NOT use when: only expanding meaning (-> wonder), only creating the initial Seed (-> socrates), only reviewing blind spots (-> evolve-step), or only defining semantic boundary (-> ontology)
  vs socrates: Socrates creates the Seed; Interview Harness continues through blind-spot review and ontology convergence.
---

# Interview Harness

## Details

Interview Harness is the master loop. It creates or reuses a Seed, evolves it through user-approved blind-spot review, then asks the user to define an ontology. It is semi-automatic because user adoption and boundary equivalence are required.

## Scratch Files

- `.symposium/scratch/socrates.md`: latest top-level `## Seed` and `## Ontology`, plus Socrates cycles.
- `.symposium/scratch/evolve-step.md`: cumulative blind-spot audit.
- `.symposium/scratch/interview-harness.md`: previous ontology snapshots and comparison logs.

## Stage 1: Seed Gate

Read `.symposium/scratch/socrates.md`.

If top-level `## Seed` is missing, or its YAML lacks `goal`, `constraints`, or `acceptance_criteria`, call `socrates`.

If all three slots exist and none are `empty`, skip `socrates`.

## Stage 2: Convergence Loop

Run 1 to 5 cycles:

1. Snapshot the current top-level `## Ontology` into `.symposium/scratch/interview-harness.md` if one exists.
2. Call `evolve-step`.
3. Call `ontology`.
4. Compare previous ontology and current ontology.
5. Stop if similarity is 0.85 or higher.
6. Stop if the cycle count exceeds 5.

## Similarity

- idea: 1 if identical, otherwise 0.
- boundary: ask the user whether the previous and current boundaries mean the same thing. 1 only if the user says yes.
- properties: Jaccard overlap, intersection size divided by union size.

Average the three scores.

First cycle has no previous ontology, so skip comparison and continue.

## Output

Return:

```yaml
final_seed:
  goal: ...
  constraints:
    - ...
  acceptance_criteria:
    - ...
  ontology:
    idea: ...
    boundary: ...
    properties:
      - ...
      - ...
      - ...
cycles: <N>
final_similarity: <0.0-1.0>
stop_reason: convergence | safety_valve
```

## Rules

- Call sub-skills; do not rewrite their internal process.
- Prefer the sub-skills' multiple-choice question style when asking for user decisions, while preserving free-form edit options.
- Keep only one canonical latest Seed and one canonical latest Ontology in `socrates.md`.
- Never auto-adopt evolve-step candidates.
- Never decide boundary equivalence without user confirmation.
- Do not exceed 5 cycles.
