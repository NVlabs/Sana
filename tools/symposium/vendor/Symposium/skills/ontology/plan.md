# Ontology Plan

## Purpose

Add a semantic boundary layer on top of a closed Seed.

## Success Criteria

- Reads the latest top-level `## Seed`.
- Copies `Seed.goal` exactly into `ontology.idea`.
- Collects one boundary and exactly three properties.
- Replaces top-level `## Ontology`.
- Does not ask for actions or data types.

## Boundaries

- Do not create the Seed; use `socrates`.
- Do not evolve Seed content; use `evolve-step`.
- Do not orchestrate convergence; use `interview-harness`.
