# Interview Harness Plan

## Purpose

Run the full semi-automatic convergence loop from vague request to final Seed plus Ontology.

## Success Criteria

- Runs the Seed Gate.
- Calls `socrates` only when the Seed is missing or incomplete.
- Runs `evolve-step` and `ontology` for 1 to 5 cycles.
- Computes ontology similarity.
- Confirms boundary equivalence with the user.
- Stops on convergence at 0.85 or safety valve.

## Boundaries

- Do not expand meaning only; use `wonder`.
- Do not create only the initial Seed; use `socrates`.
- Do not review blind spots only; use `evolve-step`.
- Do not define ontology only; use `ontology`.
