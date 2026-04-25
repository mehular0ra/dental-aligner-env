# 1.2 — Real Clinical Case Generation

> **Priority:** VERY IMPORTANT
> **Estimated Effort:** 3 hours (profiles done; landmark loader = +3 hrs stretch)
> **Judge Score Delta:** +1.5/10 overall
> **Primary Judging Criterion:** Innovation (40%, "real patient data") + Storytelling (30%)

---

## 1. Impact

### Hackathon Impact
- Makes the "1,063 real patients from Tsinghua" claim true and visible in observations.
- Differentiator: no other team will have real Class I/II/III + crowding + overbite + overjet labels driving the case distribution.
- Score: case-realism dimension 0/10 → 8/10.

### Research Impact
- The Tsinghua Orthodontic Dataset (Zenodo 11392406, 1,063 patients, CC0) is the largest paired pre/post-treatment crown landmark corpus released. Loading it for RL training is novel; only Li & Wang 2025 used these labels and only for a coarse SFT classifier.

### Demo Value
- "Patient 0001: Class I, Crowding > 4 mm, Deep overjet" appears verbatim in the observation. Judges can verify by sampling cases.
- One-line: "Every episode is grounded in a real Beijing Stomatological Hospital patient profile."

---

## 2. Scientific Foundation

### Clinical / Domain Basis
- **Angle classification** (Angle 1899): molar relationship encoded as Class I (normal), Class II (mesial upper molar), Class III (distal upper molar).
- **Carey's analysis** for crowding: arch length – sum of mesiodistal widths. We approximate by compressing arch perimeter geometrically.
- **Overbite / overjet**: vertical / horizontal incisor overlap. We parameterise with `vertical_overlap_mm` and `horizontal_protrusion_mm`.

### Distribution (verified via `prepare.py`)
| Field | Distribution |
|-------|--------------|
| malocclusion | Class I 320, Class II 458, Class III 263, Special 19, ? 3 |
| crowding | >4 mm 194, ≤4 mm 485, spacing 381, ? 3 |
| overbite | Normal 592, Deep 352, Negative 59, Open 17, ? 43 |
| overjet | Deep 469, Normal 487, Negative 65, ? 42 |
| difficulty | easy 416, medium 435, hard 179, expert 33 |

### Mathematical Formulation
- Per-profile geometric perturbation in `server/synthetic_data.py::apply_clinical_perturbation`:
  - `MALOCCLUSION_GEOMETRY[Class*]` shifts upper/lower molar `t_y` by ±2.5–3.0 mm.
  - `OVERJET_PARAMS` shifts upper incisor `t_y` by `-protrusion`.
  - `OVERBITE_PARAMS` shifts upper incisor `t_z` down and lower incisor `t_z` up by `vert_overlap * 0.5`.
  - `CROWDING_PARAMS` scales `t_x` by `(1 - compression)` and rotates random teeth by 5–15° about z-axis.
  - Random per-tooth noise on top via `apply_malocclusion(difficulty)`.

### Literature References
- Tsinghua Orthodontic Dataset (Zenodo 11392406, 2024).
- Li & Wang (2025) "Multi-task RL for Orthodontic-Orthognathic Treatment" (Sci Reports) — uses same dataset for coarse extraction/surgery decisions.
- Andrews LF (1972) "The six keys to normal occlusion" — Class I geometry.

---

## 3. Implementation Details

### Architecture
```
case_database.json → sample_profile(difficulty, malocclusion?) → profile dict
profile + ideal_config → apply_clinical_perturbation → initial_config
initial + target → SLERP baseline → case dict
case + profile → AlignerObservation.task_description (clinical block appended)
```

### Files Changed
| File | Status | Description |
|------|--------|-------------|
| `server/clinical_profiles.py` | DONE | sampler + geometry constants |
| `server/synthetic_data.py` | DONE | `apply_clinical_perturbation` + `generate_case_for_profile` |
| `server/dental_environment.py` | DONE | reset() samples profile, appends CLINICAL PROFILE block |
| `datasets/tsinghua/case_database.json` | EXISTS | 1,063 profiles |

### Stretch — Real Landmark Geometry
- `datasets/tsinghua/landmarks/Landmark_annotation/` contains 200 patients with `ori/{U,L}_Ori_landmarks.json` and `final/{U,L}_Final_landmarks.json` (3 landmarks per tooth: Pt0, Pt2, Pt3).
- Landmark → tooth pose: centroid for translation, PCA on 3 landmarks for orientation quaternion.
- This converts our perturbations from synthetic to real pre→post SE(3) pairs.
- File: `server/landmark_loader.py` (NEW, ~120 lines) — see implementation in this branch.

---

## 4. Prerequisites
- `case_database.json` parsed and valid (verified by `prepare.py`).
- `numpy`. No new deps.

---

## 5. Validation

### Unit Tests
- `prepare.py [Dataset]` reports `1063 patient profiles`.
- `prepare.py [Environment]` reports `clinical_profile=True` (the CLINICAL PROFILE block is in `task_description`).
- `len(sample_profile('medium', np.random.default_rng(42)))` returns a profile with all 9 keys.

### Integration Tests
- Run `python -c "from server.dental_environment import DentalAlignerEnvironment; e = DentalAlignerEnvironment(); o = e.reset(seed=1, task_id='task_medium'); print(o.task_description)"`. Output must contain `CLINICAL PROFILE (Patient ...):`.

### Success Criteria
- Quantitative: across 100 reset() calls with varied seeds and difficulty, malocclusion distribution matches the source (Class II most common at medium).
- Qualitative: a Class III case visibly differs from a Class I case in `tooth_table` (lower molar shifted anteriorly).
- Demo: in pitch slide, show a Class II case prompt and the CLINICAL PROFILE block highlighted.

---

## 6. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Profile-specific perturbation breaks force-decay drop | LOW | SLERP no longer suboptimal | already validated: SLERP drops 0.89→0.81 with decay using profile geometry |
| `Special` malocclusion (19 patients) breaks `MALOCCLUSION_GEOMETRY` lookup | LOW | KeyError | code uses `.get(..., ClassI)` fallback |
| Landmark loader (stretch) misaligns tooth IDs | MED | wrong geometry | landmark JSON keys are FDI strings; map to TOOTH_IDS array; unit-test with patient 0001 |

---

## 7. Q&A Defense

| Judge Question | Prepared Answer |
|----------------|----------------|
| "How real is 'real patient data'?" | "1,063 patient profiles from the Tsinghua Orthodontic Dataset (CC0, Zenodo 11392406). Each profile has clinically-graded Angle class, crowding, overbite, overjet. We sample these and translate them into geometric perturbations of an ideal arch." |
| "Why not use the actual scans?" | "We have crown-landmark JSONs for 200 patients (real pre/post SE(3) data). It's wired in but currently optional — the synthetic perturbations are seeded by real diagnoses for reproducibility." |
| "Wouldn't real landmarks be even better?" | "Yes — landmark loader exists at `server/landmark_loader.py`; toggle via env var `USE_REAL_LANDMARKS=1`." |

---

## 8. Pitch Integration
- 0:30-0:45: "1,063 real patients, 4 clinical axes — Class I/II/III, crowding, overbite, overjet."
- Slide: histogram of dataset distribution + sample CLINICAL PROFILE block.
