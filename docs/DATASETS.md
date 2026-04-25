# Dataset Documentation

> Last verified: 2026-04-25

This document describes all datasets available in the `datasets/` directory for the OrthoRL dental aligner environment.

---

## 1. Tsinghua 3D Orthodontic Dental Dataset

**Source:** [Zenodo 11392406](https://zenodo.org/records/11392406) | [Paper](https://www.nature.com/articles/s41597-024-04138-7)
**License:** CC0 (public domain), Data Use Agreement required
**Size:** ~8.5 GB (compressed)

### Overview

1,063 pre/post-orthodontic treatment pairs from 435 patients (ages 8-35). Contains 3D dental models (STL meshes), per-tooth landmark annotations, and clinical malocclusion labels.

### Directory Structure

```
datasets/tsinghua/
├── case_database.json              # 1,063 patient profiles with labels
├── landmarks/Landmark_annotation/  # 200 patient folders (extracted from RAR)
│   └── {patient_id}/
│       ├── ori/                    # Pre-treatment
│       │   ├── U_Ori_landmarks.json   # Upper arch landmarks + segmentation
│       │   └── L_Ori_landmarks.json   # Lower arch landmarks + segmentation
│       └── final/                  # Post-treatment
│           ├── U_Final_landmarks.json
│           └── L_Final_landmarks.json
├── Orthodontic_dental_dataset/     # STL meshes (extracted from multi-part RAR)
│   └── {patient_id}/
│       ├── ori/                    # Pre-treatment STL meshes
│       └── final/                  # Post-treatment STL meshes
├── Crowding.txt                    # Clinical labels
├── Malocclusion.txt
├── Overbite.txt
├── Overjet.txt
├── Dentition.txt
└── train-test-split.txt            # 810 train / 250 test / 3 unknown
```

### Case Database Schema (`case_database.json`)

Each entry is keyed by patient ID (e.g., `"0001"`) with fields:

| Field | Type | Description |
|-------|------|-------------|
| `patient_id` | string | 4-digit ID |
| `crowding` | string | `Crowding_above_4`, `Crowding_below_4`, `Dental_spacing` |
| `malocclusion` | string | `ClassI`, `ClassII`, `ClassIII`, `Special` |
| `overbite` | string | `Normal_overbite`, `Deep_overbite`, `Negative_overbite`, `Anterior_open_bite` |
| `overjet` | string | `Normal_overjet`, `Deep_overjet`, `Negative_overjet` |
| `dentition` | string | `Permanent`, `Mixed`, `Deciduous` |
| `split` | string | `train`, `test` |
| `difficulty_score` | int | 1-4 |
| `difficulty_level` | string | `easy`, `medium`, `hard`, `expert` |

### Label Distributions

| Label | Distribution |
|-------|-------------|
| **Malocclusion** | Class II: 458, Class I: 320, Class III: 263, Special: 19 |
| **Crowding** | Below 4mm: 485, Spacing: 381, Above 4mm: 194 |
| **Overbite** | Normal: 592, Deep: 352, Negative: 59, Open: 17 |
| **Overjet** | Normal: 487, Deep: 469, Negative: 65 |
| **Dentition** | Permanent: 1021, Mixed: 31, Deciduous: 7 |
| **Difficulty** | Easy: 416, Medium: 435, Hard: 179, Expert: 33 |
| **Split** | Train: 810, Test: 250 |

### Landmark File Format

Each JSON contains:
- `version`: `"1.0.0.1"`
- `segmentation`: Dict keyed by FDI tooth number (e.g., `"11"`, `"21"`), each containing `vertices` (list of [x,y,z] coordinates defining that tooth's surface)
- `landmarks`: List of landmark point coordinates

### Key Value for RL Environment

- **Pre/post treatment pairs** provide ground-truth target configurations
- **Clinical labels** enable condition-specific case generation
- **Per-tooth segmentation vertices** allow SE(3) pose extraction (centroid + PCA axes)
- **Difficulty scoring** supports curriculum learning

---

## 2. Open-Full-Jaw

**Source:** [GitHub](https://github.com/diku-dk/Open-Full-Jaw) | [Paper](https://arxiv.org/abs/2209.07576)
**License:** CC BY-NC-SA 4.0 (non-commercial, share-alike)
**Size:** 3.1 GB (17 patient ZIPs)

### Overview

17 patient-specific finite element models from CBCT scans. Contains mandible, maxilla, individual tooth STL meshes, periodontal ligament (PDL) models, and — critically — **teeth principal axes JSON files** with direct SE(3) pose data (centroid + rotation axes).

### Directory Structure

```
datasets/Open-Full-Jaw/
├── dataset/
│   └── Patient_{1-17}.zip          # Each contains:
│       └── Patient_N/
│           ├── input/
│           │   ├── mandible/                    # All 17 patients
│           │   │   ├── bone/mandible.stl        # Mandibular bone mesh
│           │   │   ├── teeth/
│           │   │   │   ├── all_teeth.obj        # Combined mesh
│           │   │   │   └── tooth_{18-31}.stl    # Individual teeth (Universal numbering)
│           │   │   └── teeth_axes_mandible.json  # SE(3) pose data
│           │   └── maxilla/                     # 12 of 17 patients
│           │       ├── bone/maxilla.stl
│           │       ├── teeth/
│           │       │   └── tooth_{2-15}.stl
│           │       └── teeth_axes_maxilla.json
│           └── output/
│               ├── mandible/
│               │   ├── parameters_log.json
│               │   ├── simulation/              # FEBio simulation files
│               │   ├── surface_meshes/          # bone.stl, pdls.stl, teeth.stl
│               │   └── volumetric_meshes/       # .msh, .vtk tetrahedral meshes
│               └── maxilla/                     # (same structure)
├── src/                             # Pipeline code
└── docs/
```

### Teeth Axes JSON Schema (SE(3) Pose Data)

Each `teeth_axes_{mandible|maxilla}.json` is a dict keyed by Universal tooth number:

```json
{
  "18": {
    "c": [90.97, 73.65, 38.17],   // centroid (x, y, z)
    "x": [91.09, 74.55, 38.60],   // x-axis endpoint (mesiodistal)
    "y": [91.88, 73.38, 38.50],   // y-axis endpoint (buccolingual)
    "z": [90.77, 73.76, 39.15]    // z-axis endpoint (occlusogingival)
  },
  ...
}
```

The axes define a local coordinate frame per tooth: `c` is the centroid, and `x/y/z` are unit-direction endpoints. Together they form an SE(3) pose (translation = c, rotation = matrix from normalized x/y/z directions).

### Summary Statistics

| Metric | Value |
|--------|-------|
| Total patients | 17 |
| Patients with mandible | 17/17 |
| Patients with maxilla | 12/17 |
| Total individual tooth STLs | 414 |
| Avg teeth per patient | 24.4 |
| Data per patient | Input meshes + output FEM simulations |

### Key Value for RL Environment

- **Direct SE(3) pose data** — no segmentation pipeline needed, ready for environment state representation
- **Individual tooth STLs** — enable collision detection and per-tooth manipulation
- **PDL meshes** — support biomechanical feasibility scoring
- **FEM simulation outputs** — validate force/displacement relationships
- **Universal tooth numbering** — maps directly to environment's tooth indexing

---

## 3. Bits2Bites

**Source:** [DITTO Lab, Univ. of Modena](https://ditto.ing.unimore.it/bits2bites/) | Paper: Borghi et al. (2025), ODIN Workshop at MICCAI 2025
**License:** Registration + license agreement required
**Size:** 1.9 GB (200 paired scans)

### Overview

200 patients, each with paired upper + lower intraoral 3D scans (400 STL meshes total) and multi-label occlusion classification annotations. Scans are in RAS (Right-Anterior-Superior) coordinate system.

### Directory Structure

```
datasets/bits2bites/
└── Bits2Bites/
    ├── Annotations.csv              # Occlusion labels for all 200 patients
    ├── 1/
    │   ├── upper.stl                # Upper arch scan (~8-10 MB)
    │   └── lower.stl                # Lower arch scan (~8-10 MB)
    ├── 2/
    │   ├── upper.stl
    │   └── lower.stl
    └── ... (through 200/)
```

### Annotations Schema (`Annotations.csv`)

| Column | Values | Description |
|--------|--------|-------------|
| `Patient` | 1-200 | Patient ID |
| `Right Class` | Class I (90), Class II Edge to Edge (51), Class II Full (32), Class III (21), Unknown (6) | Right molar Angle classification |
| `Left Class` | Class I (88), Class II Edge to Edge (51), Class II Full (28), Class III (28), Unknown (5) | Left molar Angle classification |
| `Anterior Bite` | Normal (81), Deep Bite (73), Open Bite (42), Inverted Bite (4) | Anterior vertical relationship |
| `Transversal Bite` | Normal (140), Cross Bite (56), Scissor Bite (4) | Transversal relationship (may include specific tooth numbers) |
| `Median Lines` | Deviated (122), Centered (78) | Midline alignment |

### Mesh Statistics (Sample)

| Metric | Value |
|--------|-------|
| Avg triangles per mesh | ~170,000 |
| Avg vertices per mesh | ~85,000 |
| Avg file size | 8-10 MB per STL |
| Coordinate system | RAS (Right-Anterior-Superior) |

### Key Value for RL Environment

- **Paired upper/lower arches** — enable occlusion evaluation between arches
- **Multi-label occlusion annotations** — provide natural reward signals (Angle classification, bite type, midline)
- **Large sample size (200)** — good for training robust reward models
- **Full-arch unsegmented scans** — require segmentation for per-tooth pose extraction, but can be used directly for occlusion scoring

---

## Cross-Dataset Comparison

| Feature | Tsinghua | Open-Full-Jaw | Bits2Bites |
|---------|----------|---------------|------------|
| **Patients** | 1,063 | 17 | 200 |
| **Pre/post pairs** | Yes | No | No |
| **Per-tooth segmentation** | Yes (vertex lists) | Yes (individual STLs) | No (full arch only) |
| **SE(3) poses** | Derivable from vertices | Direct (JSON axes) | Requires segmentation |
| **Clinical labels** | Yes (5 categories) | No | Yes (occlusion only) |
| **PDL/bone models** | No | Yes | No |
| **FEM simulations** | No | Yes | No |
| **Both arches** | Yes | 12/17 patients | Yes (all 200) |
| **Tooth numbering** | FDI (11-47) | Universal (2-31) | N/A |
| **License** | CC0 + DUA | CC BY-NC-SA | Registration |

## Usage in RL Environment

### Primary workflow:
1. **Open-Full-Jaw** → Direct SE(3) state initialization (17 patients, immediate use)
2. **Tsinghua** → Large-scale case generation with clinical labels (1,063 patients, SE(3) from vertex PCA)
3. **Bits2Bites** → Occlusion reward model training (200 paired arches with classification labels)

### Tooth numbering mapping:
- **FDI** (Tsinghua): 11-18 (upper right), 21-28 (upper left), 31-38 (lower left), 41-48 (lower right)
- **Universal** (Open-Full-Jaw): 1-16 (upper right→left), 17-32 (lower left→right)
