# 1.4 — README ↔ Code Consistency Audit

> **Priority:** VERY IMPORTANT
> **Estimated Effort:** 1-2 hours
> **Judge Score Delta:** +0.5/10 (defensive — prevents credibility collapse during Q&A)
> **Primary Judging Criterion:** Pipeline (10%) — but a single discrepancy can sink Innovation/Storytelling scores.

---

## 1. Impact

### Hackathon Impact
- A judge who runs `curl /step` and sees a different reward formula than the README claims will downgrade Innovation by 1-2 points and reduce trust in everything else.
- "Numbers in the README don't match what the server returns" is a known killshot — Kube SRE Gym 2025 winners explicitly flagged spec ↔ code parity as table stakes.

### Research Impact
- Reproducibility: a faithful README is a precondition for citation-quality release.

### Demo Value
- One-line: "Every formula in the README is grep-checkable against `server/grader.py`."

---

## 2. Audit — Discrepancies Found

| # | Location (README) | Claim | Code Reality | Action |
|---|-------------------|-------|--------------|--------|
| 1 | L 178 | Hard jitter "between stages 8 and 16" | Hard-coded `jitter_stage = 12` | Update README to say "stage 12" |
| 2 | L 178 | Jitter magnitudes "0.5–1.5 mm / 5–15°" | `jitter_strength=0.2` → ~0.2 mm / ~2° | Update README to "≤0.6 mm translation / ≤2° rotation per affected tooth" |
| 3 | L 161-177 | Easy "≤2 mm, ≤15°"; Medium "≤4 mm, ≤25°"; Hard "≤6 mm, ≤35°" | Easy `(1,3) mm, (5,15)°`; Medium `(2,5) mm, (10,20)°`; Hard `(3,8) mm, (15,25)°` | Update README to match `synthetic_data.apply_malocclusion` actuals |
| 4 | L 127 | `tooth_table` dict has `initial_pose` 7-vector + `target_pose` 7-vector | Code emits `ToothPoseTableRow` with flat `current_qw … target_tz` fields | Update README to flat-field shape |
| 5 | L 129 | `arch_graph_json` "edge attributes include current inter-tooth distances" | `_build_arch_graph_json` emits adjacency only | Either implement distances or drop the claim. Drop for 1_4. |
| 6 | L 240 | Staging delta_threshold "0.05 mm or 0.5 deg" | Code uses translation-only `cum_trans > 0.1` mm | Update README to "0.1 mm cumulative translation" |
| 7 | L 240 | `R_staging = max(0, SpearmanCorr)` | Code does `(rho + 1.0) / 2.0` (linear remap [-1,1]→[0,1]) | Update README formula |
| 8 | L 248 | `R_recovery = min(1, ratio) * 0.15` | Code applies the 0.15 weight in `grade_hard`, not inside `R_recovery` | Move 0.15 out of recovery formula in README |
| 9 | L 264 | "SLERP drops ~11% on medium" | Measured 8.6% (0.8884 → 0.8121) | Update to "~9%" |
| 10 | L 387-389 | Baseline table (no force decay) | Force decay now applied | Re-measure and update with decay |
| 11 | L 364 | `HF_SPACE_URL=https://grimoors-dental-aligner-env.hf.space` | Need to verify or update to actual deployment URL | Verify via Dockerfile / spec 1_5 |
| 12 | L 130 | "baseline_trajectory_json: provides a valid reference plan the agent can refine" | Acknowledged design tension (gives the agent the answer). Tier-2 cleanup. | DEFER — remains in obs for now, note it's a SLERP hint not the optimum since force decay breaks SLERP |
| 13 | (gap) | README does not mention CLINICAL PROFILE block in `task_description` | Code appends Clinical Profile block at reset() | Add a paragraph documenting it |
| 14 | (gap) | README does not mention pharmacokinetic force decay is applied **before** grading | Code: `actual = apply_force_decay(planned, init); reward = grade(actual, …)` | Make this explicit in Reward Function section |

---

## 3. Implementation Details

### Files Changed
| File | Change Type | Description |
|------|------------|-------------|
| `README.md` | MODIFY | Apply 14 fixes from table above |
| `server/grader.py` | NO CHANGE | code is the source of truth — README adapts to it |

### Approach
- README adapts to current code (faster, lower-risk than rewriting grader to match prose).
- Where a future redesign is desired (e.g., #5 inter-tooth distances), document the gap and defer.

---

## 4. Prerequisites
- 1_2 (clinical profiles), 1_3 (force decay) — both must be DONE so the README can describe the actual reward path.

---

## 5. Validation

### Unit Tests
- `grep` test: every formula in README appears verbatim somewhere in `server/grader.py` or `server/force_decay.py`.

### Integration Tests
- Reset env, apply SLERP baseline, grade — printed reward matches README's "Baseline Scores" table within ±0.02.

### Success Criteria
- Quantitative: 0 unaddressed discrepancies from the audit table.
- Qualitative: a fresh reader can predict the reward returned by `/step` from the README alone.
- Demo: in pitch, point at the README and say "every number here was just verified by `prepare.py`."

---

## 6. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| New audit discrepancies introduced by 1_1/1_3 changes | MED | partial fix | re-run `prepare.py` after every commit; run audit table again pre-submission |
| Removing baseline_trajectory_json breaks reference inference.py | MED | infra failure | DEFER removal; only update the description |

---

## 7. Q&A Defense

| Judge Question | Prepared Answer |
|----------------|----------------|
| "Why is the easy max-translation 2 mm in the README but I see 3 mm cases?" | "Documented now: `apply_malocclusion('easy')` samples in [1, 3) mm. README updated." |
| "What's the actual force-decay drop on medium?" | "8.6% relative (0.8884 → 0.8121) — measured by `prepare.py` on a fixed seed." |

---

## 8. Pitch Integration
- Indirectly: zero awkward moments when judges actually test the API.
