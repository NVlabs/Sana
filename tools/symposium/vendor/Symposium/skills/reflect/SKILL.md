---
name: reflect
description: |
  Compare meaning candidates from Wonder with the user's intended meaning, one pair at a time. In the Socrates loop, Reflect writes only the current cycle's ### Reflect section.
  Triggers (EN): reflect, compare meanings, surface meaning gap, check what I meant
  Do NOT use when: expanding meanings (-> wonder), merging meanings into shared meaning (-> refine), restating a goal (-> restate), or running the full Seed loop (-> socrates)
  vs wonder: Wonder expands candidates; Reflect checks where each candidate matches or diverges from the user's intention.
---

# Reflect

## Details

Reflect takes the active cycle's `### Wonder` candidates and asks the user whether each one matches what they meant.

Ask one comparison at a time. Do not decide which meaning is correct. Do not invent missing user intent.

## Question Style

Prefer multiple-choice checks so the user can classify each meaning quickly. Always allow a free-form correction.

Use this shape:

```markdown
Does this candidate match what you meant?
Candidate: <meaning candidate>
A. Yes, this is close
B. Partly, but the nuance is different
C. No, this is not what I meant
D. Other / correct the nuance in your own words
```

Use the user's selected or written nuance in the `User's intended nuance` field.

## Output

Write a comparison list:

```markdown
- Meaning A
  - User's intended nuance: ...
  - Model's interpreted nuance: ...
  - Same / different: ...
```

## Scratch Contract

Read from `.symposium/scratch/socrates.md` under the active cycle's `### Wonder`.

Write to the same cycle's `### Reflect`.

When every meaning candidate has a visible same/different comparison, stop and hand off to `refine`.
