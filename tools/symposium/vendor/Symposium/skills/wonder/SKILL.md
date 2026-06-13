---
name: wonder
description: |
  Expand a vague word or request by asking what other meanings may be hidden inside it. In the Socrates loop, Wonder writes only the current cycle's ### Wonder section.
  Triggers (EN): wonder, expand meaning, explore hidden meanings, what else could this mean
  Do NOT use when: comparing meaning gaps (-> reflect), merging meanings (-> refine), restating a goal (-> restate), or running the full Seed loop (-> socrates)
  vs reflect: Wonder expands meaning candidates; Reflect compares those candidates against the user's intended meaning.
---

# Wonder

## Details

Wonder expands the hidden meanings inside the user's vague request until at least three user-grounded meaning candidates are visible.

Ask the user one question at a time. Use `request_user_input` in Codex Plan mode when available, `AskUserQuestion` in Claude when available, or a normal conversational question otherwise. Do not invent meanings to fill the section.

Do not complete Wonder immediately after generating multiple-choice options. Options are probes, not collected meanings. A meaning candidate is collected only after the user selects, rejects, edits, or explains it.

## Word Boundary Discovery

Wonder is about the meaning of the user's words. Keep asking what the user's key word or phrase means in this context until its boundary becomes clear.

Before asking, identify the smallest quoted phrase that carries the ambiguity. Ask about that phrase's meaning, not about implementation choices or next actions.

For the first Wonder question, infer several possible semantic boundaries around that word or phrase, then ask which boundary is closest.

Do not use a fixed question template. Read the user's wording and generate boundary candidates from what the phrase could mean in context:

- the smallest concrete artifact the phrase could refer to
- the larger workflow, decision, or operating model the phrase may imply
- the personal, organizational, or strategic tacit knowledge hidden behind the phrase

The choices should differ by where the meaning begins and ends, not by surface variants of the same meaning. Surface variants, implementation details, and taxonomic subtypes can be asked later after the boundary is clear.

Each follow-up must stay attached to the same word or phrase by asking one of:

- What does this word include here?
- What does this word exclude here?
- Which nearby meaning sounds similar but would be wrong?
- What unstated assumption makes this word obvious to the user?

Do not switch to project planning, feature selection, stack selection, or acceptance criteria while still in Wonder.

## Question Style

Prefer multiple-choice prompts so the user can answer quickly, but treat choices as hypotheses, not truth. Always include an open edit option.

Use this shape:

```markdown
Which hidden meaning is closest to what you meant?
A. <candidate meaning>
B. <candidate meaning>
C. <candidate meaning>
D. Other / edit in your own words
```

If the user picks A-C, record only the meaning they selected or confirmed. If they pick D or answer freely, record their wording.

## Interview Loop

Run at least two user turns unless the user explicitly says the first answer is complete:

1. Ask a boundary-choice question.
2. After the user answers, ask one follow-up about the same word's meaning: what it includes, what it excludes, which nearby meaning is wrong, or what assumption makes the word obvious to the user.
3. Record the user's answer as meaning candidates.
4. Continue only until at least three user-grounded candidates are visible, then hand off to `reflect`.

Never treat model-generated choices alone as the final Wonder output.

## Output

Write at least three user-grounded meaning candidates as a Markdown list:

```markdown
- Meaning A — one-line explanation
- Meaning B — one-line explanation
- Meaning C — one-line explanation
```

## Scratch Contract

Write the list to `.symposium/scratch/socrates.md` under the active `## Cycle N` and `### Wonder` section.

If there is no active cycle, create:

```markdown
## User Input
<original request>

## Cycle 1
### Wonder
...
```

When at least three user-grounded meaning candidates are collected and the word boundary is clear enough to compare against the user's intent, stop and hand off to `reflect`.
