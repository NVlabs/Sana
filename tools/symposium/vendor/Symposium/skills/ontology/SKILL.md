---
name: ontology
description: |
  Add a semantic boundary layer on top of a closed Seed. Use Seed.goal as idea, ask for boundary and exactly three properties, and replace the top-level ## Ontology section in .symposium/scratch/socrates.md.
  Triggers (EN): ontology, create ontology, define semantic boundary, add meaning boundary
  Do NOT use when: no Seed exists yet (-> socrates), Seed content must evolve (-> evolve-step), or the full convergence harness is needed (-> interview-harness)
  vs evolve-step: Evolve-step changes the Seed; Ontology adds a boundary layer above the Seed.
---

# Ontology

## Details

Ontology defines what the Seed's goal means and where that meaning stops. It does not ask for actions; `acceptance_criteria` already covers actions.

## Input

Read the latest Seed from `.symposium/scratch/socrates.md` top-level `## Seed`. If there is no Seed, stop and ask the user to run `socrates` first.

## Ontology Shape

```yaml
ontology:
  idea: <Seed.goal exactly>
  boundary: <what is inside and outside the idea>
  properties:
    - <property 1>
    - <property 2>
    - <property 3>
```

## Process

1. Copy `Seed.goal` exactly into `ontology.idea`.
2. Ask one boundary question: "Where does this idea's meaning begin and end?"
3. Ask for exactly three properties inside that boundary.
4. Replace the top-level `## Ontology` section in `.symposium/scratch/socrates.md`.

Use `request_user_input` in Codex Plan mode when available, `AskUserQuestion` in Claude when available, or normal conversational questions otherwise.

## Question Style

Prefer multiple-choice prompts for both boundary and properties, but keep them editable. Choices are candidate boundaries/properties, not final ontology terms.

Boundary prompt shape:

```markdown
Where does this idea's meaning begin and end?
A. <boundary candidate>
B. <boundary candidate>
C. <boundary candidate>
D. Other / define the boundary in your own words
```

Properties prompt shape:

```markdown
Which three properties must be inside that boundary?
A. <property>, <property>, <property>
B. <property>, <property>, <property>
C. <property>, <property>, <property>
D. Other / list exactly three properties
```

Only write terms the user selected or wrote.

## Rules

- `idea` and `boundary` are different.
- Every property must be inside the boundary.
- Exactly three properties are required.
- Do not ask for data types.
- Do not ask for actions.
- Only user-provided boundary and property terms enter the YAML.

## Convergence Comparison

The `interview-harness` compares the previous ontology and the current ontology:

- idea: 1 if identical, otherwise 0.
- boundary: 1 only if the user confirms the two boundaries mean the same thing.
- properties: Jaccard overlap between the two property sets.

Average the three scores. A score of 0.85 or higher means converged.
