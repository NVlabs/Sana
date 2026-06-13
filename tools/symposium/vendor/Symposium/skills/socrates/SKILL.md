---
name: socrates
description: |
  Run Wonder, Reflect, Refine, and Restate as a Ralph-style loop to turn one vague request into a closed Seed with goal, constraints, and acceptance_criteria. The latest Seed is stored in .symposium/scratch/socrates.md under the top-level ## Seed section.
  Triggers (EN): socrates, run socrates, create seed, clarify vague request, close this request into a seed
  Do NOT use when: a Seed already exists and only blind spots need review (-> evolve-step), convergence through ontology is needed (-> interview-harness), or only one meaning-expansion step is needed (-> wonder)
  vs interview-harness: Socrates creates the Seed; Interview Harness continues with evolve-step and ontology convergence.
---

# Socrates

## Details

Socrates orchestrates four sub-skills: `wonder`, `reflect`, `refine`, and `restate`. It does not rewrite their internals; it calls them in order and checks whether the Seed is closed.

## Scratch Contract

Use `.symposium/scratch/socrates.md`.

Cycles accumulate:

```markdown
## User Input
<original request>

## Cycle 1
### Wonder
...
### Reflect
...
### Refine
shared meaning: ...
### Restate
goal: ...
```

The canonical latest Seed is always a top-level section:

````markdown
## Seed
> Source: /socrates Cycle N

```yaml
goal: ...
constraints:
  - ...
acceptance_criteria:
  - ...
```
````

Replace the top-level `## Seed` section on each closed Seed. Do not append multiple canonical Seeds.

## Seed Gate

After each cycle, check:

- `goal`: current cycle's `### Restate` has one goal.
- `constraints`: at least one user-stated constraint or boundary exists.
- `acceptance_criteria`: at least one verifiable criterion exists.

If all three are filled and the user says the current goal is not materially different from the previous cycle's goal, stop.

If a slot is empty, narrow the next Wonder question toward that slot:

- Missing goal: "What result do you actually want from this request?"
- Missing constraints: "What must not happen, or what boundary must not be crossed?"
- Missing acceptance criteria: "How will you know this result is good enough?"

Ask narrowed questions as multiple choice when possible. Provide 3 plausible options and one `Other / edit in your own words` option. The options are only scaffolding; do not fill Seed slots unless the user selects, confirms, or writes the content.

Do not fill empty slots by guessing.

## Safety Valve

Maximum 5 cycles. If a slot is still missing after 5 cycles, write `empty` for that slot and stop.

## Output

The final output is the top-level `## Seed` YAML block.
