---
name: restate
description: |
  Restate the shared meaning from Refine as one executable goal sentence. In the Socrates loop, Restate writes only the current cycle's ### Restate section.
  Triggers (EN): restate, restate goal, one-line goal, turn this into a goal
  Do NOT use when: expanding meanings (-> wonder), comparing meanings (-> reflect), creating shared meaning (-> refine), or running the full Seed loop (-> socrates)
  vs refine: Refine creates shared meaning; Restate turns that shared meaning into a goal another person can execute.
---

# Restate

## Details

Restate turns the active cycle's `### Refine` shared meaning into one goal sentence that another person could read and pursue.

Ask the user to write or approve one goal candidate. Do not freely write the goal for the user.

## Question Style

Prefer multiple-choice goal drafts, plus an edit option. A selected draft is not final until it clearly preserves the user's shared meaning.

Use this shape:

```markdown
Which goal statement should become the Seed goal?
A. <goal candidate>
B. <goal candidate>
C. <goal candidate>
D. Other / rewrite it in your own words
```

If the user chooses A-C and the choice is clear, write that goal. If they choose D or answer freely, use their wording.

## Output

Write one line:

```markdown
goal: <one user-approved goal sentence>
```

## Scratch Contract

Read from `.symposium/scratch/socrates.md` under the active cycle's `### Refine`.

Write to the same cycle's `### Restate`.

When the user confirms the goal, stop. The `socrates` orchestrator will build the Seed.
