---
title: Dental Aligner Trajectory Planning Environment
emoji: 🦷
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - dental
  - orthodontics
  - trajectory-planning
  - reinforcement-learning
  - se3
  - medical-ai
---

# Dental Aligner Trajectory Planning Environment — battisiBot

## Overview

Orthodontic treatment with clear aligners (e.g., Invisalign) requires a clinician — or increasingly, an AI planning system — to divide the total corrective movement of each tooth into a discrete sequence of small steps. Each step corresponds to one physical aligner tray that the patient wears for approximately two weeks.

### Why This Matters (Real-World Impact)
Automated orthodontic treatment planning is a $4B+ industry (Invisalign generates ~$4B/year). Current systems require manual planning by orthodontists — an AI agent that can generate clinically valid 24-stage aligner trajectories would:
- Reduce planning time from 2-3 hours per case to seconds
- Enable same-day treatment planning
- Standardize quality across practitioners
- Power the next generation of automated aligner fabrication systems

This environment directly benchmarks the capabilities needed for such an agent:
- **Spatial reasoning in SE(3)**: Can the agent reason about 3D rotations and translations?
- **Long-horizon planning**: 24 sequential decisions with compounding effects
- **Clinical constraint satisfaction**: Hard limits on per-stage movement
- **Robustness to perturbation**: Recover from adversarial jitter (patient non-compliance)
- **Staged priority reasoning**: Incisors move before molars — does the agent know orthodontic clinical order?

**Why 24-stage trajectory planning is non-trivial**

Each of the 28 teeth occupies a pose in SE(3): three translational degrees of freedom and three rotational. A 24-stage plan therefore defines a path through a (28 × 6)-dimensional configuration space. Several constraints make naive interpolation insufficient:

- **Clinical staging priority** — incisors and canines typically move earlier in treatment; molars and premolars act as anchorage units and move later. Violating this order risks anchorage loss, root resorption, or treatment regression.
- **Smoothness** — adjacent stages must differ by at most ~0.25 mm translation and ~2° rotation per tooth. Larger per-stage deltas exceed the elastic range of the aligner material and cause patient discomfort.
- **Constraint compliance** — teeth must not physically collide, roots must remain within alveolar bone, and the inter-arch relationship (overjet/overbite) must monotonically converge toward the target.
- **SE(3) geometry** — rotations are non-commutative; naive linear interpolation of Euler angles produces gimbal-lock artifacts and incorrect intermediate poses. Correct interpolation requires quaternion SLERP or Lie-group geodesics.

**Why this matters for RL / agents**

- **Long-horizon planning**: 24 steps with sparse terminal reward.
- **SE(3) spatial reasoning**: actions live on a non-Euclidean manifold.
- **Adversarial robustness**: the hard task injects mid-trajectory jitter (simulating patient non-compliance) and requires recovery planning.
- **Structured output generation**: the agent must produce a numerically valid JSON trajectory at each step.

**battisiBot**

battisiBot is the reference hybrid agent for this environment. It combines:
1. An LLM reasoning pass that identifies which teeth to prioritize at each stage and generates a staging plan in natural language.
2. A Python post-processing pass that applies SLERP interpolation anchored to the LLM's staging schedule, enforces per-step delta budgets, and resolves any collision constraints via a geometric solver.

The name derives from *battisi* (बत्तीसी), the Hindi word for 32 — referencing the full adult dentition.

---

## Tooth Pose Representation

Each tooth is represented as a 7-vector:

```
[qw, qx, qy, qz, tx, ty, tz]
```

| Component | Meaning | Units |
|-----------|---------|-------|
| qw, qx, qy, qz | Unit quaternion encoding tooth orientation | dimensionless |
| tx | Left-right translation (positive = patient's right) | mm |
| ty | Anterior-posterior translation (positive = anterior) | mm |
| tz | Superior-inferior translation (positive = superior) | mm |

**Unit quaternion constraint**: the quaternion must satisfy `qw² + qx² + qy² + qz² = 1`. Violating this constraint produces undefined rotations. The environment normalises submitted quaternions before scoring, but agents should submit unit quaternions to avoid silent rescaling.

**FDI tooth numbering** — 28-tooth arch layout (third molars excluded):

```
               UPPER ARCH (viewed from above)
  Left side ←                            → Right side
  (patient's left)                    (patient's right)

   17  16  15  14  13  12  11 | 21  22  23  24  25  26  27
   ──────────────────────────────────────────────────────
   47  46  45  44  43  42  41 | 31  32  33  34  35  36  37

               LOWER ARCH (viewed from above)

  Quadrant 1: 11-17 (upper right)    Quadrant 2: 21-27 (upper left)
  Quadrant 3: 31-37 (lower left)     Quadrant 4: 41-47 (lower right)

  Position key: x1=central incisor, x2=lateral incisor, x3=canine,
                x4=first premolar,  x5=second premolar,
                x6=first molar,     x7=second molar
```

**Grounding in real research**: the SE(3) pose representation used in this environment mirrors the format of large-scale intraoral scan datasets. Landmark-free deep learning methods for orthodontic planning (e.g., work built on 280K intraoral scan corpora) represent each tooth as a rigid body pose in SE(3), making this environment directly relevant to current research directions in AI-assisted orthodontics.

---

## Action Space

The agent submits an `AlignerAction` at each step.

| Field | Type | Description |
|-------|------|-------------|
| `trajectory` | `list[list[list[float]]]` | Shape (26, 28, 7). Full trajectory from stage 0 (initial) through stage 25 (final). Each inner list is a 7-vector `[qw, qx, qy, qz, tx, ty, tz]`. Stage 0 must match the observed initial configuration; stage 25 must match the target. |
| `reasoning` | `str` | Free-text explanation of the agent's planning decisions. Used for interpretability logging; not scored. |
| `confidence` | `float` | Agent's self-assessed confidence in the plan, in `[0.0, 1.0]`. Used for logging; not scored. |

---

## Observation Space

The agent receives an `AlignerObservation` at each step.

| Field | Type | Description |
|-------|------|-------------|
| `current_stage` | `int` | Current stage index (0–25). Always 0 at reset. |
| `stages_remaining` | `int` | Number of stages left including the current one. |
| `task_id` | `str` | One of `"task_easy"`, `"task_medium"`, `"task_hard"`. |
| `tooth_table` | `list[dict]` | Structured list of dicts, one per tooth. Each dict contains `tooth_id`, `initial_pose` (7-vector), and `target_pose` (7-vector). |
| `tooth_table_text` | `str` | Human-readable formatted table of tooth poses. |
| `arch_graph_json` | `str` | JSON string encoding the dental arch as a graph. Nodes are teeth; edges connect adjacent teeth in each arch and cross-arch antagonist pairs. Edge attributes include current inter-tooth distances. |
| `baseline_trajectory_json` | `str` | JSON string of the SLERP baseline trajectory, shape (26, 28, 7). Provides a valid reference plan the agent can refine. |
| `adversarial_jitter_applied` | `bool` | True if mid-trajectory jitter has been applied (task_hard only). |
| `jitter_description` | `str \| None` | Natural-language description of which teeth were jittered and by how much. None if no jitter applied. |
| `last_plan_feedback` | `str \| None` | Structured feedback on the most recent submitted plan (constraint violations, smoothness score, staging order issues). None on the first step. |
| `step_number` | `int` | Number of agent steps taken so far in the current episode. |

---

## State

The `AlignerState` is the internal environment state (not directly observed by the agent, but logged for evaluation).

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Active task identifier. |
| `initial_config` | `np.ndarray` | Shape (28, 7). Initial malocclusion tooth poses. |
| `target_config` | `np.ndarray` | Shape (28, 7). Target (corrected) tooth poses. |
| `perturbed_tooth_ids` | `list[int]` | FDI IDs of teeth that differ between initial and target. |
| `current_trajectory` | `np.ndarray \| None` | Shape (26, 28, 7). The agent's most recently submitted trajectory. |
| `baseline_trajectory` | `np.ndarray` | Shape (26, 28, 7). SLERP reference trajectory computed at reset. |
| `jitter_config` | `dict \| None` | Parameters of the injected jitter (affected teeth, magnitude, stage). None if no jitter. |
| `step_count` | `int` | Number of agent steps taken. |
| `done` | `bool` | True when the episode has terminated. |

---

## Task Descriptions

**task_easy**

- 4–6 randomly selected teeth are perturbed from a baseline arch.
- Perturbation magnitudes: translation ≤ 2 mm, rotation ≤ 15°.
- Malocclusion class: Class I crowding (teeth displaced within arch, no skeletal discrepancy).
- No adversarial jitter.
- Scoring emphasis: final accuracy (40%), smoothness (20%), compliance (20%), staging quality (20%).

**task_medium**

- 10–14 teeth perturbed.
- Perturbation magnitudes: translation ≤ 4 mm, rotation ≤ 25°.
- Malocclusion class: Class II (upper arch protrusive relative to lower; anterior-posterior correction required).
- No adversarial jitter.
- Scoring emphasis: final accuracy (45%), smoothness (25%), constraint compliance (15%), staging quality (15%).

**task_hard**

- 18–24 teeth perturbed.
- Perturbation magnitudes: translation ≤ 6 mm, rotation ≤ 35°.
- Adversarial jitter injected mid-trajectory (between stages 8 and 16): selected teeth are displaced by 0.5–1.5 mm / 5–15° to simulate patient non-compliance (wearing aligners irregularly).
- The agent is notified of the jitter via `adversarial_jitter_applied` and `jitter_description` in the observation.
- A recovery bonus rewards agents that successfully re-route from the jittered state to the target.
- Scoring emphasis: final accuracy (40%), smoothness (20%), constraint compliance (15%), staging quality (10%), recovery bonus (15%).

---

## Reward Function

All reward components are normalised to `[0, 1]` before weighting. The final episode reward is a weighted sum.

### Final Accuracy

Per-tooth score combining translational and rotational error:

```
trans_error_i  = ||t_pred_i - t_target_i||_2          (mm)
rot_error_i    = 2 * arccos(|q_pred_i · q_target_i|)  (radians)

tooth_score_i  = exp(-trans_error_i / 3.0) * exp(-rot_error_i / 0.5)

R_accuracy     = (1 / N) * sum_i( w_i * tooth_score_i )
```

where `N = 28`, `w_i` is a per-tooth clinical weight (incisors = 1.0, canines = 1.2, premolars = 0.9, molars = 0.8), and the dot product `q_pred_i · q_target_i` is clamped to `[-1, 1]` before arccos.

### Smoothness

Measures consistency of per-step deltas across the trajectory:

```
delta_trans_{i,s}  = ||t_{i,s+1} - t_{i,s}||_2       for s in 0..24
delta_rot_{i,s}    = 2 * arccos(|q_{i,s+1} · q_{i,s}|)

combined_delta_{i,s} = delta_trans_{i,s} + alpha * delta_rot_{i,s}
                       (alpha = 3.0 / pi converts radians to mm-equivalent)

variance_i   = Var_s( combined_delta_{i,s} )

R_smoothness = exp( -mean_i( variance_i ) / sigma_target )
               (sigma_target = 0.05)
```

### Constraint Compliance

Binary-style penalty based on the fraction of (tooth, stage) pairs that satisfy:

```
delta_trans <= budget_trans     (budget_trans = 0.25 mm)
delta_rot   <= budget_rot       (budget_rot   = 2 * pi/180 * 2 = 0.035 rad)

R_compliance = fraction of (tooth, stage) pairs where both budgets are met
```

### Staging Quality

Spearman rank correlation between the agent's staging schedule and the clinically preferred order (incisors first, then canines, then premolars, then molars):

```
preferred_order_i  = clinical_priority(tooth_i)   (lower = earlier)
agent_stage_i      = first stage s where delta_{i,s} > delta_threshold
                     (delta_threshold = 0.05 mm or 0.5 deg)

R_staging = max(0, SpearmanCorr(preferred_order, agent_stage))
```

### Recovery Bonus (task_hard only)

```
post_jitter_error  = mean_i( ||t_{i,final} - t_{i,target}||_2 )
baseline_error     = mean_i( ||t_{i,initial} - t_{i,target}||_2 )

recovery_ratio     = 1 - (post_jitter_error / baseline_error)

R_recovery = clip(recovery_ratio, 0, 1)
```

### Component Weight Tables

**task_easy**

| Component | Weight |
|-----------|--------|
| R_accuracy | 0.40 |
| R_smoothness | 0.20 |
| R_compliance | 0.20 |
| R_staging | 0.20 |
| R_recovery | 0.00 |

**task_medium**

| Component | Weight |
|-----------|--------|
| R_accuracy | 0.45 |
| R_smoothness | 0.25 |
| R_compliance | 0.15 |
| R_staging | 0.15 |
| R_recovery | 0.00 |

**task_hard**

| Component | Weight |
|-----------|--------|
| R_accuracy | 0.40 |
| R_smoothness | 0.20 |
| R_compliance | 0.15 |
| R_staging | 0.10 |
| R_recovery | 0.15 |

---

## Setup and Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install uv first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install all dependencies from the lockfile:

```bash
uv sync
```

This creates a `.venv` automatically and installs all pinned dependencies from `uv.lock`.

---

## Running Locally

Start the server:

```bash
uv run uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Verify health:

```bash
curl http://localhost:7860/health
```

Start a new episode:

```bash
curl -X POST http://localhost:7860/reset
```

Submit an action:

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "trajectory": [...],
    "reasoning": "Prioritising upper incisors in stages 1-8.",
    "confidence": 0.72
  }'
```

Request a visualization GIF (base64-encoded):

```bash
curl -X POST http://localhost:7860/visualize \
  -H "Content-Type: application/json" \
  -d '{"trajectory": [...]}'
```

---

## Running the Inference Agent

Set the required environment variables and run:

```bash
export HF_TOKEN=<your-hf-token>
export HF_SPACE_URL=https://grimoors-dental-aligner-env.hf.space
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

uv run python inference.py
```

Expected stdout format (evaluated by the grader):

```
[START] task=task_easy env=dental-aligner-env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action='plan 24 stages for 28 teeth' reward=0.91 done=true error=null
[END] success=true steps=1 score=0.91 rewards=0.91
```

---

## Baseline Scores

Scores are single-episode rewards measured against the live HF Space.

| Task | SLERP Baseline | battisiBot (LLM-planned) | Notes |
|------|---------------|--------------------------|-------|
| task_easy | ~0.87 | ~0.91 | LLM prioritises incisors first |
| task_medium | ~0.89 | ~0.85 | Staging quality contributes 15% |
| task_hard | ~0.32 | ~0.34 | Recovery bonus up to +15% |

The SLERP baseline applies quaternion spherical linear interpolation uniformly across all 24 stages without any clinical staging logic. It achieves moderate smoothness but low staging quality scores.

battisiBot's gain over SLERP comes primarily from:
- LLM-guided staging order (incisors move first, molars last).
- Per-step delta budget enforcement in the Python post-processing pass.
- Recovery re-planning after jitter detection (task_hard).

---

## Visualization

The environment includes a GIF animation module (`server/visualization.py`) that renders dental arch trajectories as animated top-down views.

**Features:**
- One frame per aligner stage (stages 0–25, giving 26 frames total).
- 3-frame pause at the start (initial configuration) and end (final configuration) for easy inspection.
- Upper and lower arches rendered as separate subplots side-by-side.
- Each tooth is drawn as an oriented ellipse sized to anatomical proportions (molars wider, incisors narrower).
- A white arrow inside each ellipse indicates the tooth's current orientation (yaw extracted from quaternion).
- FDI tooth ID labels are printed inside each ellipse.
- Color coding: upper arch in blue (`#4A90D9`), lower arch in orange (`#E8894A`).
- Comparison mode (`generate_comparison_gif`) shows SLERP baseline and agent trajectory side-by-side in a 2×2 subplot grid for direct visual comparison.
- Base64 export (`trajectory_to_gif_base64`) enables embedding GIFs directly in API responses without intermediate files.

**Usage example:**

```python
from server.visualization import trajectory_to_gif, generate_comparison_gif

# Save a single trajectory GIF
trajectory_to_gif(agent_trajectory, "agent_plan.gif", fps=4)

# Save a side-by-side comparison GIF
generate_comparison_gif(baseline_trajectory, agent_trajectory, "comparison.gif", fps=4)
```
