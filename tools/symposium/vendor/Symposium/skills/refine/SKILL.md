---
name: refine
description: |
  Merge meaning gaps from Reflect into one user-approved shared meaning. In the Socrates loop, Refine writes only the current cycle's ### Refine section.
  Triggers (EN): refine, create shared meaning, merge meanings, consolidate intent
  Do NOT use when: expanding meanings (-> wonder), comparing meaning gaps (-> reflect), restating a goal (-> restate), or running the full Seed loop (-> socrates)
  vs reflect: Reflect exposes differences; Refine turns the accepted differences into one shared meaning.
---

# Refine

## Details

Refine takes the active cycle's `### Reflect` comparisons and asks the user to merge the useful parts into one shared meaning.

Ask for one candidate shared meaning at a time. If the user does not approve it, keep asking; do not write a consensus sentence by guesswork.

## Question Style

Prefer offering two or three candidate shared meanings, plus an edit option. Choices are draft language only; the shared meaning must be user-approved.

Use this shape:

```markdown
Which shared meaning should we keep?
A. <shared meaning candidate>
B. <shared meaning candidate>
C. <shared meaning candidate>
D. Other / rewrite it in your own words
```

If the user chooses A-C, ask for confirmation only when the choice still appears ambiguous. If they choose D or answer freely, use their wording.

## Output

Write one line:

```markdown
shared meaning: <one user-approved sentence>
```

## Scratch Contract

Read from `.symposium/scratch/socrates.md` under the active cycle's `### Reflect`.

Write to the same cycle's `### Refine`.

When the user approves the shared meaning, stop and hand off to `restate`.
