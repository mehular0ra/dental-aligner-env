# Feature Spec Format — OrthoRL Hackathon

Each spec file follows this structure. Sections marked [REQUIRED] must be filled. Sections marked [IF RELEVANT] are included when applicable.

---

```markdown
# {Tier}.{Number} — {Feature Name}

> **Priority:** VERY IMPORTANT / IMPORTANT / GOOD TO HAVE / STRETCH
> **Estimated Effort:** X hours
> **Judge Score Delta:** +X.X/10 overall
> **Primary Judging Criterion:** Innovation (40%) / Storytelling (30%) / Rewards (20%) / Pipeline (10%)

---

## 1. Impact [REQUIRED]

### Hackathon Impact
- Which judging criteria does this address and by how much?
- What score does this take us FROM → TO?
- How does this compare to what SF winners had?

### Research Impact
- Does this fill a gap in the literature?
- Is this publishable?
- Who would use this?

### Demo Value
- How does this show up in the 3-minute pitch?
- What's the one-line pitch for this feature?
- Visual artifact produced (GIF, plot, screenshot)?

---

## 2. Scientific Foundation [REQUIRED]

### Clinical / Domain Basis
- What clinical principle does this implement?
- What is the orthodontic/medical justification?
- Key measurements, thresholds, and parameters with citations

### Mathematical Formulation
- Equations, algorithms, or formal definitions
- How existing SE(3) / quaternion math connects
- Reward function changes (exact formulas)

### Literature References
- Primary papers (with specific sections/figures cited)
- Datasets used
- How this differs from prior work

---

## 3. Implementation Details [REQUIRED]

### Architecture
- Which module(s) are modified or created?
- How does this integrate with existing StepwiseDentalEnvironment?
- Data flow diagram (input → processing → output)

### Files Changed
| File | Change Type | Description |
|------|------------|-------------|
| path/to/file.py | NEW / MODIFY / DELETE | What changes |

### Code Sketch
- Pseudocode or actual Python for core logic
- Function signatures with types
- Key data structures

### API Changes [IF RELEVANT]
- New endpoints
- Modified observation/action schemas
- Backward compatibility notes

---

## 4. Prerequisites [REQUIRED]

### Dependencies
- Other features that must be built first (by spec ID)
- Python packages needed
- Data files needed

### Assumptions
- What must be true for this to work?
- What are we assuming about the environment state?

---

## 5. Validation [REQUIRED]

### Unit Tests
- Specific test cases with expected inputs/outputs
- Edge cases to verify

### Integration Tests
- End-to-end verification steps (curl commands, scripts)
- How to verify this works with the rest of the system

### Success Criteria
- Quantitative: "SLERP score drops from 0.87 to 0.55 with force decay"
- Qualitative: "Observation includes clinical_profile field"
- Demo: "GIF shows trained agent deviating from SLERP at stage 8"

---

## 6. Risks & Mitigations [REQUIRED]

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Description | LOW/MED/HIGH | What breaks | How to handle |

---

## 7. Q&A Defense [REQUIRED]

| Judge Question | Prepared Answer |
|----------------|----------------|
| "Why did you...?" | "Because..." |

---

## 8. Pitch Integration [IF RELEVANT]

- Which 15-second segment of the 3-minute pitch does this appear in?
- What visual/slide accompanies it?
- What's the "wow" moment?
```

---

## Naming Convention

`{tier}_{number}_{snake_case_name}.md`

Examples:
- `1_1_grpo_training_pipeline.md`
- `1_3_pharmacokinetic_force_decay.md`
- `2_2_clinical_knowledge_exam.md`
- `3_1_rotating_expert_panel.md`
