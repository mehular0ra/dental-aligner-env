# Dataset Findings & Insights for Agent Use

> Generated: 2026-04-25 | For use by agents building/improving the OrthoRL dental aligner environment
> Reference: See `docs/DATASETS.md` for full schema and structure documentation

---

## Quick Reference: What's Available

| Dataset | Patients | Per-tooth poses | Pre/Post pairs | Clinical labels | Arches |
|---------|----------|----------------|----------------|-----------------|--------|
| Tsinghua | 1,063 | Via vertex PCA | Yes | 5 categories | Upper + Lower |
| Open-Full-Jaw | 17 | Direct JSON (c,x,y,z) | No | No | Mandible + Maxilla (12/17) |
| Bits2Bites | 200 | No (full arch only) | No | Occlusion (5 cols) | Upper + Lower |

---

## Patient-Level Deep Dives

### Tsinghua Patient 0001

**Clinical profile:**
```json
{
  "crowding": "Crowding_above_4",
  "malocclusion": "ClassI",
  "overbite": "Normal_overbite",
  "overjet": "Deep_overjet",
  "dentition": "Permanent",
  "difficulty_level": "medium"
}
```

**Data available:**
- 4 STL meshes: U_Ori (156K tri), L_Ori (180K tri), U_Final (137K tri), L_Final (157K tri)
- 4 JSON files with per-tooth vertex segmentation (FDI numbering)
- 4 landmark files with vertex lists per tooth + landmark points
- Pre-treatment upper: 14 teeth (FDI 11-17, 21-27), 75,498 vertices total
- Post-treatment upper: 12 teeth (2 extracted during treatment)
- Pre-treatment lower: 14 teeth, 87,546 vertices total

**Treatment displacement analysis (upper arch):**

| Tooth (FDI) | Displacement | Direction |
|-------------|-------------|-----------|
| 11 (central incisor) | 4.97 mm | Primarily anterior (+Y), slight buccal |
| 12 (lateral incisor) | 3.86 mm | Anterior + slight mesial |
| 13 (canine) | 6.00 mm | Strong anterior, some extrusion |
| 15 (2nd premolar) | 0.71 mm | Minor adjustment |
| 16 (1st molar) | 0.05 mm | Virtually stationary (anchor) |

**Insight:** Molars serve as anchors (near-zero displacement) while anterior teeth show large movements (3-6mm). This matches clinical reality and provides ground-truth for staging: posterior teeth should move first/least, anterior teeth move most over more stages.

### Open-Full-Jaw Patient 3

**Data available:**
- Mandible: 14 teeth (Universal 18-31), individual STLs (40K-100K triangles each)
- Maxilla: 14 teeth (Universal 2-15), individual STLs
- Teeth axes JSON: centroid + 3 unit axes per tooth (SE(3) pose)
- Bone STLs: mandible.stl, maxilla.stl
- PDL surface meshes: pdls.stl (14.7 MB mandible, 18.0 MB maxilla)
- Volumetric FEM meshes: .msh + .vtk (37-49 MB each)
- FEBio simulation files: tipping simulation configs

**SE(3) pose example (mandible tooth 18 — right 3rd molar):**
```
centroid: [29.59, 11.83, 8.96]
axes: x/y/z all unit magnitude (1.0000)
  x-axis: mesiodistal direction
  y-axis: buccolingual direction
  z-axis: occlusogingival direction
```

**Mesh resolution per tooth:**
- Molars: 90K-100K triangles (4.5-5.0 MB) — highest detail
- Premolars: 40K-65K triangles (2.0-3.2 MB)
- Incisors: ~50K triangles (2.5 MB)

**Insight:** The axes JSON provides **ready-to-use SE(3) poses** — no PCA/segmentation pipeline needed. Each tooth has a unit orthonormal frame at its centroid. This is the most direct representation for the environment's state space.

### Bits2Bites Patient 1

**Annotations:**
```
Right Class: Class I
Left Class: Class I
Anterior Bite: Normal
Transversal Bite: Normal
Median Lines: Deviated
```

**Mesh statistics:**
- Upper arch: 170,411 triangles, bbox 39.7 x 45.8 x 14.3 mm
- Lower arch: 170,798 triangles, bbox 55.0 x 46.8 x 12.3 mm
- Upper centroid: (-12.58, 12.72, 5.41)
- Lower centroid: (10.17, 10.05, -8.18)
- Vertical separation (Z): ~13.6 mm between arch centroids

**Insight:** Full-arch scans without per-tooth segmentation. The occlusion labels map directly to Andrews' criteria used in the reward function. The centroid positions and bounding boxes reveal the RAS coordinate convention and can calibrate the environment's spatial scale.

---

## Key Insights for Environment Development

### 1. SE(3) State Extraction Pipeline

Three tiers of pose data availability:

| Tier | Source | Method | Ready? |
|------|--------|--------|--------|
| **Immediate** | Open-Full-Jaw `teeth_axes_*.json` | Direct read | Yes |
| **Compute** | Tsinghua vertex segmentation | PCA on per-tooth vertices → centroid + axes | ~10 lines of code |
| **Pipeline** | Bits2Bites full-arch STL | Need MeshSegNet/DentalSegmentator first | No |

**Code to extract SE(3) from Tsinghua vertices:**
```python
import numpy as np
vertices = np.array(segmentation[tooth_id]["vertices"])  # Nx3
centroid = vertices.mean(axis=0)  # translation
centered = vertices - centroid
cov = np.cov(centered.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
rotation_matrix = eigenvectors[:, ::-1]  # largest variance first
# SE(3) = (centroid, rotation_matrix)
```

### 2. Treatment Trajectory Ground Truth

Tsinghua is the **only dataset with pre/post pairs**. Key findings:

- **Tooth extraction during treatment:** Some teeth present in pre but absent in post (e.g., Patient 0001: 14→12 upper teeth). The environment must handle variable tooth counts.
- **Displacement magnitude ranges:** 0-6mm translation per tooth across full treatment. Typical per-stage limit should be 0.25-0.5mm (matching clinical 2-week aligner intervals over 24 stages ≈ 48 weeks).
- **Anchor teeth pattern:** Molars (FDI 16, 26, 36, 46) show near-zero displacement. The environment should penalize excessive molar movement.
- **Anterior teeth move most:** Incisors and canines show the largest displacements (3-6mm). Crowding resolution concentrates here.

### 3. Occlusion Reward Calibration

Bits2Bites provides 200 cases with clinician-labeled Angle classifications:

- **Class I (normal):** 90/200 right, 88/200 left — baseline "good occlusion"
- **Class II (distal):** 83/200 right, 79/200 left — common malocclusion
- **Class III (mesial):** 21/200 right, 28/200 left — less common
- **Deep bite:** 73/200 — vertical excess
- **Open bite:** 42/200 — vertical deficiency
- **Cross bite:** 56/200 — transversal deviation

These distributions can train a classifier to score occlusion from arch geometry, providing a learned reward component that supplements the analytical Andrews' metrics already in the environment.

### 4. Coordinate System Mapping

| Dataset | System | Convention |
|---------|--------|-----------|
| Tsinghua | Custom | X=lateral, Y=anterior-posterior, Z=vertical |
| Open-Full-Jaw | Custom | Varies by patient CBCT orientation |
| Bits2Bites | RAS | X=Right, Y=Anterior, Z=Superior |

**Action required:** The environment should normalize all inputs to a consistent coordinate frame. Recommend using the dental convention: X=mesiodistal, Y=buccolingual, Z=occlusogingival, centered at arch midpoint.

### 5. Tooth Numbering Translation

```python
# FDI (Tsinghua) ↔ Universal (Open-Full-Jaw) mapping
FDI_TO_UNIVERSAL = {
    # Upper right: FDI 11-18 → Universal 8-1
    11: 8, 12: 7, 13: 6, 14: 5, 15: 4, 16: 3, 17: 2, 18: 1,
    # Upper left: FDI 21-28 → Universal 9-16
    21: 9, 22: 10, 23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16,
    # Lower left: FDI 31-38 → Universal 24-17
    31: 24, 32: 23, 33: 22, 34: 21, 35: 20, 36: 19, 37: 18, 38: 17,
    # Lower right: FDI 41-48 → Universal 25-32
    41: 25, 42: 26, 43: 27, 44: 28, 45: 29, 46: 30, 47: 31, 48: 32,
}
```

### 6. Biomechanical Data (Open-Full-Jaw only)

- **PDL meshes:** 14.7-18.0 MB per arch — can validate the Kelvin-Voigt PDL model already in the environment
- **FEBio simulation configs:** Contain material properties, boundary conditions, and loading scenarios for tipping simulations
- **Volumetric meshes:** Enable finite element validation of force-displacement relationships
- **`parameters_log.json`:** Contains the pipeline parameters used to generate each patient's FEM model

### 7. Scale and Resolution Benchmarks

| Metric | Tsinghua | Open-Full-Jaw | Bits2Bites |
|--------|----------|---------------|------------|
| Triangles per arch | 137K-180K | ~900K (sum of teeth) | ~170K |
| Vertices per tooth | 3,800-5,400 | ~16K-33K | N/A |
| Arch width (mm) | ~50 | ~60 | 40-55 |
| Tooth STL size | N/A (combined) | 2-5 MB each | N/A (combined) |

### 8. Data Loading Priority for Agents

1. **For environment state initialization:** Use Open-Full-Jaw `teeth_axes_*.json` — zero preprocessing, direct SE(3)
2. **For case diversity and curriculum:** Use Tsinghua `case_database.json` — 1,063 labeled cases with difficulty levels
3. **For reward model training:** Use Bits2Bites `Annotations.csv` + arch pairs — 200 classified cases
4. **For treatment trajectory targets:** Use Tsinghua pre/post landmark pairs — ground truth displacements
5. **For biomechanical validation:** Use Open-Full-Jaw PDL + FEM outputs — real tissue models

---

## File Paths Quick Reference

```python
DATASET_ROOT = "datasets/"

# Tsinghua
TSINGHUA_CASES     = f"{DATASET_ROOT}tsinghua/case_database.json"
TSINGHUA_LANDMARKS = f"{DATASET_ROOT}tsinghua/landmarks/Landmark_annotation/{{patient_id}}/{{phase}}/{{arch}}_landmarks.json"
TSINGHUA_STL       = f"{DATASET_ROOT}tsinghua/Orthodontic_dental_dataset/{{patient_id}}/{{phase}}/{{arch}}.stl"
TSINGHUA_STL_JSON  = f"{DATASET_ROOT}tsinghua/Orthodontic_dental_dataset/{{patient_id}}/{{phase}}/{{arch}}.json"

# Open-Full-Jaw (need to unzip Patient_N.zip first)
OFJ_AXES           = f"{DATASET_ROOT}Open-Full-Jaw/dataset/Patient_{{n}}/input/{{arch}}/teeth_axes_{{arch}}.json"
OFJ_TOOTH_STL      = f"{DATASET_ROOT}Open-Full-Jaw/dataset/Patient_{{n}}/input/{{arch}}/teeth/tooth_{{num}}.stl"
OFJ_BONE_STL       = f"{DATASET_ROOT}Open-Full-Jaw/dataset/Patient_{{n}}/input/{{arch}}/bone/{{arch}}.stl"
OFJ_PDL_STL        = f"{DATASET_ROOT}Open-Full-Jaw/dataset/Patient_{{n}}/output/{{arch}}/surface_meshes/pdls.stl"

# Bits2Bites
B2B_ANNOTATIONS    = f"{DATASET_ROOT}bits2bites/Bits2Bites/Annotations.csv"
B2B_UPPER_STL      = f"{DATASET_ROOT}bits2bites/Bits2Bites/{{patient_id}}/upper.stl"
B2B_LOWER_STL      = f"{DATASET_ROOT}bits2bites/Bits2Bites/{{patient_id}}/lower.stl"
```

---

## Disk Usage Summary

| Dataset | Compressed | Extracted | Status |
|---------|-----------|-----------|--------|
| Tsinghua labels/landmarks | 389 MB | ~500 MB | Extracted |
| Tsinghua STL meshes | 7.5 GB (4 RAR parts) | Extracting | In progress |
| Tsinghua ZIP archive | 7.9 GB | — | Can delete after extraction |
| Open-Full-Jaw | 3.1 GB (17 ZIPs) | ~6 GB unzipped | ZIPs verified, extract on demand |
| Bits2Bites | 1.9 GB | ~4 GB | Extracted |
| **Total** | **~13 GB** | **~17 GB** | |
