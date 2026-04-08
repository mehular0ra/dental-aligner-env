"""
battisiBot — Dental Aligner Trajectory Planning Inference Agent
battisi = 32 in Hindi (a full set of teeth)

Hybrid LLM + Python approach:
  1. LLM outputs interpolation parameters per tooth (start_stage, end_stage, ease_in, ease_out, priority)
  2. Python computes the exact trajectory numerically using SLERP + easing functions
  3. This guarantees valid quaternions and respects clinical constraints

External dependencies: openai only (numpy and requests replaced with stdlib)

Usage:
  API_BASE_URL=... MODEL_NAME=... HF_TOKEN=... HF_SPACE_URL=... python inference.py
"""

import json
import math
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN")
HF_SPACE_URL = os.environ.get("HF_SPACE_URL", "http://localhost:7860")

# Benchmark identifier used in structured stdout logs
BENCHMARK = "dental-aligner-env"

# ---------------------------------------------------------------------------
# Clinical constants (mirrored from dental_constants.py)
# ---------------------------------------------------------------------------
TOOTH_IDS = [
    11, 12, 13, 14, 15, 16, 17,   # upper right
    21, 22, 23, 24, 25, 26, 27,   # upper left
    31, 32, 33, 34, 35, 36, 37,   # lower left
    41, 42, 43, 44, 45, 46, 47,   # lower right
]
N_TEETH = 28
N_STAGES = 24
MAX_TRANSLATION_PER_STAGE_MM = 0.25
MAX_ROTATION_PER_STAGE_DEG = 2.0

TOOTH_TYPES = {
    11: "central_incisor",  12: "lateral_incisor",  13: "canine",
    14: "premolar_1",       15: "premolar_2",        16: "molar_1",
    17: "molar_2",
    21: "central_incisor",  22: "lateral_incisor",  23: "canine",
    24: "premolar_1",       25: "premolar_2",        26: "molar_1",
    27: "molar_2",
    31: "central_incisor",  32: "lateral_incisor",  33: "canine",
    34: "premolar_1",       35: "premolar_2",        36: "molar_1",
    37: "molar_2",
    41: "central_incisor",  42: "lateral_incisor",  43: "canine",
    44: "premolar_1",       45: "premolar_2",        46: "molar_1",
    47: "molar_2",
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are battisiBot, an expert orthodontic treatment planning AI for the dental aligner trajectory planner.

TOOTH POSE FORMAT: Each tooth is a 7-vector [qw, qx, qy, qz, tx, ty, tz]
  - qw, qx, qy, qz = unit quaternion rotation (MUST satisfy qw²+qx²+qy²+qz²=1)
  - tx, ty, tz = translation in millimetres from the dental arch origin

BASELINE APPROACH (SLERP):
  For stage k (alpha = k/25): q_k = slerp(q_initial, q_target, alpha), t_k = lerp(t_initial, t_target, alpha)
  This naive baseline scores ~0.40. Your job is to beat it.

CLINICAL CONSTRAINTS:
  - Max translation per tooth per stage: 0.25 mm
  - Max rotation per tooth per stage: 2.0 degrees
  - All quaternions must remain unit quaternions at every stage

STAGING STRATEGY (critical for higher scores):
  - Move incisors EARLY (start_stage 1-4): teeth 11,12,21,22,31,32,41,42
  - Move canines NEXT (start_stage 3-7): teeth 13,23,33,43
  - Move premolars MIDDLE (start_stage 6-14): teeth 14,15,24,25,34,35,44,45
  - Move molars LAST (start_stage 12-20): teeth 16,17,26,27,36,37,46,47
  - Teeth that are already near target (remaining_trans_mm < 0.5) can use start_stage=1, end_stage=24 uniformly

OUTPUT FORMAT: Respond ONLY with valid JSON (no markdown, no explanation):
{
  "tooth_plans": [
    {
      "tooth_id": 11,
      "start_stage": 2,
      "end_stage": 22,
      "ease_in": 0.3,
      "ease_out": 0.3,
      "priority": "early"
    }
    // ... 27 more entries for all 28 teeth
  ],
  "reasoning": "Brief description of interpolation strategy",
  "confidence": 0.75
}"""


# ---------------------------------------------------------------------------
# HTTP helper (stdlib only — no requests)
# ---------------------------------------------------------------------------

def _http_post(url: str, payload: Dict, timeout: int = 60) -> Dict:
    """POST JSON payload and return parsed JSON response. Raises on HTTP error."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Math utilities (pure Python — no numpy)
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _vec_norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def ease_inout(t: float, ease_in: float, ease_out: float) -> float:
    """
    Blend between linear and smoothstep for eased interpolation.
    t in [0, 1], returns eased t in [0, 1].
    ease_in / ease_out each in [0, 1].
    """
    smooth = t * t * (3.0 - 2.0 * t)   # smoothstep
    result = (1 - ease_in) * t + ease_in * smooth
    result = (1 - ease_out) * result + ease_out * smooth
    return _clamp(result, 0.0, 1.0)


def quaternion_normalize(q: List[float]) -> List[float]:
    """Normalize quaternion to unit length. Returns [qw, qx, qy, qz]."""
    n = _vec_norm(q)
    if n < 1e-10:
        return [1.0, 0.0, 0.0, 0.0]
    return [x / n for x in q]


def quaternion_slerp(q0: List[float], q1: List[float], t: float) -> List[float]:
    """
    Spherical linear interpolation between unit quaternions q0 and q1.
    t=0 returns q0, t=1 returns q1.  Convention: [qw, qx, qy, qz].
    """
    q0 = quaternion_normalize(q0)
    q1 = quaternion_normalize(q1)

    dot = sum(a * b for a, b in zip(q0, q1))

    # Ensure shortest arc
    if dot < 0.0:
        q1 = [-x for x in q1]
        dot = -dot

    dot = _clamp(dot, -1.0, 1.0)

    if dot > 0.9995:
        # Nearly identical — linear interpolation
        result = [a + t * (b - a) for a, b in zip(q0, q1)]
        return quaternion_normalize(result)

    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return quaternion_normalize([s0 * a + s1 * b for a, b in zip(q0, q1)])


def quaternion_multiply(q1: List[float], q2: List[float]) -> List[float]:
    """Hamilton product q1 * q2.  Both [qw, qx, qy, qz]."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ]


def quaternion_inverse(q: List[float]) -> List[float]:
    """Quaternion inverse = conjugate (assumes unit quaternion)."""
    return [q[0], -q[1], -q[2], -q[3]]


def quaternion_to_angle_deg(q: List[float]) -> float:
    """Return rotation angle in degrees.  Uses 2 * arccos(|qw|)."""
    q = quaternion_normalize(q)
    qw = _clamp(q[0], -1.0, 1.0)
    return math.degrees(2.0 * math.acos(abs(qw)))


# ---------------------------------------------------------------------------
# Trajectory computation
# ---------------------------------------------------------------------------

def compute_tooth_trajectory(
    initial_pose: List[float],
    target_pose: List[float],
    start_stage: int,
    end_stage: int,
    ease_in: float,
    ease_out: float,
    n_stages: int = 24,
) -> List[List[float]]:
    """
    Compute a single tooth's interpolated trajectory across n_stages stages.

    Returns a list of n_stages poses (stages 1..n_stages), each a
    [qw, qx, qy, qz, tx, ty, tz] list.

    - Stages before start_stage : pose = initial_pose
    - Stages after  end_stage   : pose = target_pose
    - In between                : SLERP for rotation, LERP for translation,
                                  with easing applied
    """
    q_init = list(initial_pose[:4])
    t_init = list(initial_pose[4:7])
    q_tgt  = list(target_pose[:4])
    t_tgt  = list(target_pose[4:7])

    q_init = quaternion_normalize(q_init)
    q_tgt  = quaternion_normalize(q_tgt)

    # Clamp staging parameters
    start_stage = int(_clamp(start_stage, 1, n_stages))
    end_stage   = int(_clamp(end_stage, start_stage, n_stages))

    span = end_stage - start_stage + 1e-8

    trajectory: List[List[float]] = []
    for stage in range(1, n_stages + 1):
        if stage < start_stage:
            q_k = list(q_init)
            t_k = list(t_init)
        elif stage > end_stage:
            q_k = list(q_tgt)
            t_k = list(t_tgt)
        else:
            raw_alpha = (stage - start_stage) / span
            alpha = ease_inout(raw_alpha, ease_in, ease_out)
            q_k = quaternion_slerp(q_init, q_tgt, alpha)
            t_k = [(1.0 - alpha) * t_init[j] + alpha * t_tgt[j] for j in range(3)]

        q_k = quaternion_normalize(q_k)
        pose = [float(q_k[0]), float(q_k[1]), float(q_k[2]), float(q_k[3]),
                float(t_k[0]), float(t_k[1]), float(t_k[2])]
        trajectory.append(pose)

    return trajectory


def enforce_clinical_constraints(
    trajectory_stages: List[Dict],
    initial_poses: List[List[float]],
    tooth_plans: List[Dict],
) -> List[Dict]:
    """
    Clamp per-stage translations and rotations to clinical limits.

    Parameters
    ----------
    trajectory_stages : list of 24 stage dicts, each with 'poses' key
        (list of 28 poses in TOOTH_IDS order, each [qw,qx,qy,qz,tx,ty,tz])
    initial_poses : list of 28 initial poses (stage 0)
    tooth_plans : LLM plan (used only for ordering metadata; not mutated)

    Returns the same list, with poses clamped in-place.
    """
    n_stages = len(trajectory_stages)

    # Build a (n_stages+1) x N_TEETH x 7 working array including stage 0
    # arr[s][i] = pose as list of 7 floats
    arr: List[List[List[float]]] = [
        [[0.0] * 7 for _ in range(N_TEETH)]
        for _ in range(n_stages + 1)
    ]
    arr[0] = [list(p) for p in initial_poses]

    for s_idx, stage in enumerate(trajectory_stages):
        poses = stage["poses"]
        for i in range(N_TEETH):
            arr[s_idx + 1][i] = list(poses[i])

    # Forward pass: enforce limits stage-by-stage
    for s in range(1, n_stages + 1):
        prev = arr[s - 1]
        curr = [list(p) for p in arr[s]]

        for i in range(N_TEETH):
            # --- Translation delta ---
            delta_t = [curr[i][j] - prev[i][j] for j in range(4, 7)]
            dist_t = _vec_norm(delta_t)
            if dist_t > MAX_TRANSLATION_PER_STAGE_MM and dist_t > 1e-10:
                scale = MAX_TRANSLATION_PER_STAGE_MM / dist_t
                curr[i][4] = prev[i][4] + delta_t[0] * scale
                curr[i][5] = prev[i][5] + delta_t[1] * scale
                curr[i][6] = prev[i][6] + delta_t[2] * scale

            # --- Rotation delta ---
            q_prev = quaternion_normalize(prev[i][:4])
            q_curr = quaternion_normalize(curr[i][:4])
            q_rel = quaternion_multiply(q_curr, quaternion_inverse(q_prev))
            rot_deg = quaternion_to_angle_deg(q_rel)

            if rot_deg > MAX_ROTATION_PER_STAGE_DEG and rot_deg > 1e-6:
                frac = MAX_ROTATION_PER_STAGE_DEG / rot_deg
                q_curr = quaternion_slerp(q_prev, q_curr, frac)
                q_curr = quaternion_normalize(q_curr)
                curr[i][0] = q_curr[0]
                curr[i][1] = q_curr[1]
                curr[i][2] = q_curr[2]
                curr[i][3] = q_curr[3]

            # Always keep unit quaternion
            qn = quaternion_normalize(curr[i][:4])
            curr[i][0] = qn[0]
            curr[i][1] = qn[1]
            curr[i][2] = qn[2]
            curr[i][3] = qn[3]

        arr[s] = curr

    # Write back into the list structure
    for s_idx in range(n_stages):
        for i in range(N_TEETH):
            trajectory_stages[s_idx]["poses"][i] = list(arr[s_idx + 1][i])

    return trajectory_stages


# ---------------------------------------------------------------------------
# Observation parsing helpers
# ---------------------------------------------------------------------------

def _parse_tooth_table(obs_data: Dict) -> List[Dict]:
    """
    Return a list of 28 dicts, each with keys:
      tooth_id, tooth_type, current_pose, target_pose,
      remaining_trans_mm, remaining_rot_deg
    Works whether obs_data contains tooth_table (list of dicts) or
    tooth_table_text (markdown string).
    """
    rows = []

    tooth_table = obs_data.get("tooth_table") or []

    if tooth_table and isinstance(tooth_table, list) and len(tooth_table) > 0:
        first = tooth_table[0]
        if isinstance(first, dict):
            for row in tooth_table:
                # Support both old field names (current_pose/target_pose list)
                # and new split field names (current_qw … current_tz)
                if "current_pose" in row and "target_pose" in row:
                    current_pose = row["current_pose"]
                    target_pose  = row["target_pose"]
                else:
                    current_pose = [
                        row.get("current_qw", 1.0),
                        row.get("current_qx", 0.0),
                        row.get("current_qy", 0.0),
                        row.get("current_qz", 0.0),
                        row.get("current_tx", 0.0),
                        row.get("current_ty", 0.0),
                        row.get("current_tz", 0.0),
                    ]
                    target_pose = [
                        row.get("target_qw", 1.0),
                        row.get("target_qx", 0.0),
                        row.get("target_qy", 0.0),
                        row.get("target_qz", 0.0),
                        row.get("target_tx", 0.0),
                        row.get("target_ty", 0.0),
                        row.get("target_tz", 0.0),
                    ]

                dist_mm  = row.get("remaining_trans_mm", row.get("dist_mm",  0.0))
                dist_deg = row.get("remaining_rot_deg",  row.get("dist_deg", 0.0))

                rows.append({
                    "tooth_id":           row["tooth_id"],
                    "tooth_type":         row.get("tooth_type", TOOTH_TYPES.get(row["tooth_id"], "unknown")),
                    "current_pose":       current_pose,
                    "target_pose":        target_pose,
                    "remaining_trans_mm": dist_mm,
                    "remaining_rot_deg":  dist_deg,
                })
            return rows

    # Fallback: build minimal rows from tooth_table_text (we cannot parse poses
    # from markdown easily, so return empty-pose stubs — the caller will use
    # baseline_trajectory_json instead of per-tooth data)
    tooth_table_text = obs_data.get("tooth_table_text", "")
    if tooth_table_text:
        for tid in TOOTH_IDS:
            rows.append({
                "tooth_id":           tid,
                "tooth_type":         TOOTH_TYPES.get(tid, "unknown"),
                "current_pose":       [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "target_pose":        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "remaining_trans_mm": 0.0,
                "remaining_rot_deg":  0.0,
            })
    return rows


def _extract_initial_target_poses(
    obs_data: Dict,
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Return (initial_poses, target_poses) as lists of 28 × 7 floats.
    Tries tooth_table first; falls back to baseline_trajectory_json for
    stage 0 (initial) and stage 25 approximation (target).
    """
    rows = _parse_tooth_table(obs_data)

    # Check if we actually got real pose data
    has_real_data = rows and any(
        abs(r["current_pose"][4]) > 1e-6 or abs(r["target_pose"][4]) > 1e-6
        for r in rows
    )

    if has_real_data and len(rows) == N_TEETH:
        initial_poses = [r["current_pose"] for r in rows]
        target_poses  = [r["target_pose"]  for r in rows]
        return initial_poses, target_poses

    # Fallback: use baseline trajectory
    baseline_json = obs_data.get("baseline_trajectory_json", "")
    if baseline_json:
        try:
            baseline = json.loads(baseline_json)
            stage_1  = baseline.get("1", [])
            stage_24 = baseline.get("24", [])
            if stage_1 and stage_24:
                return stage_1, stage_24
        except (json.JSONDecodeError, KeyError):
            pass

    # Last resort: identity poses
    identity = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return [identity] * N_TEETH, [identity] * N_TEETH


# ---------------------------------------------------------------------------
# User message builder
# ---------------------------------------------------------------------------

def build_user_message(obs_data: Dict, task_id: str, stage: int = 0) -> str:
    """
    Build a rich text prompt for battisiBot describing the current episode state.
    """
    rows = _parse_tooth_table(obs_data)
    difficulty_map = {
        "task_easy": "easy",
        "task_medium": "medium",
        "task_hard": "hard",
    }
    difficulty = difficulty_map.get(task_id, "easy")

    lines: List[str] = []
    lines.append(f"=== battisiBot Dental Planning Request ===")
    lines.append(f"Task ID : {task_id}  |  Difficulty : {difficulty}  |  Current stage : {stage}")
    lines.append("")

    # --- Grader weights ---
    lines.append("GRADER WEIGHTS:")
    if task_id == "task_easy":
        lines.append("  final_accuracy=50%  smoothness=25%  compliance=25%")
    elif task_id == "task_medium":
        lines.append("  final_accuracy=45%  smoothness=20%  compliance=20%  staging_quality=15%")
    elif task_id == "task_hard":
        lines.append("  final_accuracy=40%  smoothness=15%  compliance=15%  staging_quality=15%  recovery=15%")
        lines.append("  NOTE: adversarial jitter will be applied after step 1; recovery quality matters!")
    lines.append("")

    # --- Tooth table ---
    lines.append("TOOTH TABLE (28 teeth, FDI numbering):")
    header = (
        f"{'Tooth':>5}  {'Type':<18}  {'CurrTx':>7} {'CurrTy':>7} {'CurrTz':>7}  "
        f"{'TgtTx':>7} {'TgtTy':>7} {'TgtTz':>7}  {'Dist_mm':>8} {'Dist_deg':>9}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for row in rows:
        cp = row["current_pose"]
        tp = row["target_pose"]
        lines.append(
            f"{row['tooth_id']:>5}  {row['tooth_type']:<18}  "
            f"{cp[4]:>7.2f} {cp[5]:>7.2f} {cp[6]:>7.2f}  "
            f"{tp[4]:>7.2f} {tp[5]:>7.2f} {tp[6]:>7.2f}  "
            f"{row['remaining_trans_mm']:>8.3f} {row['remaining_rot_deg']:>9.3f}"
        )
    lines.append("")

    # --- Worked example (first 3 perturbed teeth) ---
    perturbed = sorted(rows, key=lambda r: -r["remaining_trans_mm"])[:3]
    if perturbed:
        lines.append("SLERP BASELINE WORKED EXAMPLE (for reference):")
        for row in perturbed:
            tid = row["tooth_id"]
            cp  = row["current_pose"]
            tp  = row["target_pose"]
            alpha_1 = 1.0 / 25.0
            t_k = [(1 - alpha_1) * cp[4 + j] + alpha_1 * tp[4 + j] for j in range(3)]
            lines.append(
                f"  Tooth {tid}: stage-1 SLERP translation = "
                f"({t_k[0]:.3f}, {t_k[1]:.3f}, {t_k[2]:.3f})  "
                f"[remaining: {row['remaining_trans_mm']:.2f} mm, {row['remaining_rot_deg']:.1f} deg]"
            )
        lines.append("")

    # --- Staging hints based on movement distances ---
    lines.append("RECOMMENDED STAGING (based on movement distances):")
    incisor_ids  = [11, 12, 21, 22, 31, 32, 41, 42]
    canine_ids   = [13, 23, 33, 43]
    premolar_ids = [14, 15, 24, 25, 34, 35, 44, 45]
    molar_ids    = [16, 17, 26, 27, 36, 37, 46, 47]

    row_map = {r["tooth_id"]: r for r in rows}
    for group_name, group_ids, hint_start, hint_end in [
        ("Incisors",  incisor_ids,  1,  10),
        ("Canines",   canine_ids,   3,  14),
        ("Premolars", premolar_ids, 6,  18),
        ("Molars",    molar_ids,    12, 24),
    ]:
        tid_infos = []
        for tid in group_ids:
            if tid in row_map:
                r = row_map[tid]
                tid_infos.append(f"{tid}({r['remaining_trans_mm']:.1f}mm)")
        lines.append(
            f"  {group_name:<10}: start={hint_start}-{hint_start+2}, end={hint_end}-{hint_end+2}  "
            + " ".join(tid_infos)
        )
    lines.append("")

    # --- Constraint reminders ---
    lines.append("CONSTRAINTS (HARD LIMITS — violations reduce compliance score):")
    lines.append(f"  Max translation per tooth per stage : {MAX_TRANSLATION_PER_STAGE_MM} mm")
    lines.append(f"  Max rotation per tooth per stage    : {MAX_ROTATION_PER_STAGE_DEG} deg")
    lines.append(f"  All quaternions must be unit length (qw²+qx²+qy²+qz²=1)")
    lines.append(f"  Output exactly {N_TEETH} tooth_plans for tooth IDs: {TOOTH_IDS}")
    lines.append("")

    # --- Teeth near target (no movement needed) ---
    near_target = [r for r in rows if r["remaining_trans_mm"] < 0.5 and r["remaining_rot_deg"] < 3.0]
    if near_target:
        near_ids = [r["tooth_id"] for r in near_target]
        lines.append(
            f"TEETH NEAR TARGET (dist < 0.5 mm & < 3 deg, use uniform staging): {near_ids}"
        )
        lines.append("")

    lines.append("Respond ONLY with the JSON object as specified in your system prompt.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_battisibot(
    client: OpenAI,
    user_msg: str,
    max_retries: int = 3,
) -> Dict:
    """
    Call the LLM and parse its JSON response.

    Returns a dict with keys: tooth_plans, reasoning, confidence.
    Falls back to a uniform SLERP plan on repeated failure.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    last_error = ""
    raw_text = ""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.2,
                max_tokens=4096,
            )
            raw_text = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
            if raw_text.startswith("```"):
                lines = raw_text.split("\n")
                start = 1 if lines[0].startswith("```") else 0
                end   = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
                raw_text = "\n".join(lines[start:end]).strip()

            parsed = json.loads(raw_text)

            # Validate structure
            if "tooth_plans" not in parsed:
                raise ValueError("Missing 'tooth_plans' key in LLM response")
            if len(parsed["tooth_plans"]) == 0:
                raise ValueError("Empty tooth_plans list")

            # Ensure all 28 teeth are present; fill missing with defaults
            plan_map = {p["tooth_id"]: p for p in parsed["tooth_plans"]}
            full_plans = []
            for tid in TOOTH_IDS:
                if tid in plan_map:
                    plan = plan_map[tid]
                    plan["start_stage"] = int(_clamp(plan.get("start_stage", 1), 1, 24))
                    plan["end_stage"]   = int(_clamp(plan.get("end_stage",   24), plan["start_stage"], 24))
                    plan["ease_in"]     = float(_clamp(plan.get("ease_in",   0.3), 0.0, 1.0))
                    plan["ease_out"]    = float(_clamp(plan.get("ease_out",  0.3), 0.0, 1.0))
                else:
                    plan = {
                        "tooth_id":    tid,
                        "start_stage": 1,
                        "end_stage":   24,
                        "ease_in":     0.2,
                        "ease_out":    0.2,
                        "priority":    "uniform",
                    }
                full_plans.append(plan)

            return {
                "tooth_plans": full_plans,
                "reasoning":   parsed.get("reasoning", ""),
                "confidence":  float(parsed.get("confidence", 0.5)),
            }

        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            last_error = str(exc)
            if attempt < max_retries - 1:
                messages.append({"role": "assistant", "content": raw_text})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your previous response had a parse error: {last_error}\n"
                        "Please respond with ONLY valid JSON matching the required format. "
                        "No markdown, no explanation — pure JSON starting with '{' and ending with '}'."
                    ),
                })
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries - 1:
                time.sleep(2)

    # All retries exhausted — return a safe default (uniform SLERP)
    print(f"  [WARN] battisiBot LLM failed after {max_retries} attempts: {last_error}")
    print(f"  [WARN] Falling back to clinical-priority default staging.")
    return _default_tooth_plans()


def _default_tooth_plans() -> Dict:
    """
    Fallback plan when LLM fails: clinical-priority staging without easing.
    Incisors early, molars late.
    """
    staging = {
        "central_incisor": (1,  10),
        "lateral_incisor": (2,  12),
        "canine":          (4,  14),
        "premolar_1":      (7,  18),
        "premolar_2":      (8,  19),
        "molar_1":         (13, 23),
        "molar_2":         (14, 24),
    }
    plans = []
    for tid in TOOTH_IDS:
        tt = TOOTH_TYPES.get(tid, "central_incisor")
        start, end = staging.get(tt, (1, 24))
        plans.append({
            "tooth_id":    tid,
            "start_stage": start,
            "end_stage":   end,
            "ease_in":     0.3,
            "ease_out":    0.3,
            "priority":    "default",
        })
    return {
        "tooth_plans": plans,
        "reasoning":   "Default clinical-priority fallback (LLM unavailable)",
        "confidence":  0.4,
    }


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(
    client: OpenAI,
    task_id: str,
    space_url: str,
) -> Tuple[float, int]:
    """
    Run a single evaluation task end-to-end.

    Returns (total_reward, n_steps).
    """
    space_url = space_url.rstrip("/")

    # --- Reset ---
    reset_payload = {
        "task_id":    task_id,
        "model_name": f"battisiBot-{MODEL_NAME}",
    }
    reset_data = _http_post(f"{space_url}/reset", reset_payload, timeout=60)

    obs_data  = reset_data.get("observation", {})
    diff_label = task_id.replace("task_", "")

    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    total_reward  = 0.0
    n_steps       = 0
    step_rewards: List[float] = []

    # ---- Step 1: plan 24 stages ----
    user_msg = build_user_message(obs_data, task_id, stage=0)
    llm_result = call_battisibot(client, user_msg)

    tooth_plans  = llm_result["tooth_plans"]
    reasoning    = llm_result["reasoning"]
    confidence   = llm_result["confidence"]
    plan_map     = {p["tooth_id"]: p for p in tooth_plans}

    initial_poses, target_poses = _extract_initial_target_poses(obs_data)

    # Compute trajectories for all 28 teeth
    all_tooth_trajectories: List[List[List[float]]] = []
    for i, tid in enumerate(TOOTH_IDS):
        plan = plan_map.get(tid, {
            "start_stage": 1, "end_stage": 24, "ease_in": 0.3, "ease_out": 0.3
        })
        traj = compute_tooth_trajectory(
            initial_pose=initial_poses[i],
            target_pose=target_poses[i],
            start_stage=plan["start_stage"],
            end_stage=plan["end_stage"],
            ease_in=plan["ease_in"],
            ease_out=plan["ease_out"],
            n_stages=N_STAGES,
        )
        all_tooth_trajectories.append(traj)

    # Transpose: all_tooth_trajectories[tooth_idx][stage_idx]
    # -> trajectory_stages[stage_idx][tooth_idx]
    trajectory_stages: List[Dict] = []
    for s in range(N_STAGES):
        stage_poses = [all_tooth_trajectories[i][s] for i in range(N_TEETH)]
        trajectory_stages.append({
            "stage_index": s + 1,
            "tooth_ids":   TOOTH_IDS,
            "poses":       stage_poses,
        })

    # Enforce clinical constraints
    trajectory_stages = enforce_clinical_constraints(
        trajectory_stages, initial_poses, tooth_plans
    )

    # Build action payload and submit step 1
    action_payload = {
        "action": {
            "trajectory": trajectory_stages,
            "reasoning":  reasoning,
            "confidence": confidence,
            "metadata":   {},
        }
    }

    step_data = _http_post(f"{space_url}/step", action_payload, timeout=120)

    n_steps += 1
    reward_1 = step_data.get("reward")
    done_1   = step_data.get("done", True)

    # reward can be None on task_hard step 1 (jitter applied, episode continues)
    if reward_1 is None:
        reward_1 = 0.0

    total_reward = float(reward_1)
    step_rewards.append(total_reward)
    print(
        f"[STEP] step={n_steps} action='plan 24 stages for 28 teeth' "
        f"reward={total_reward:.2f} done={str(done_1).lower()} error=null",
        flush=True,
    )

    # ---- Step 2 (task_hard only): handle adversarial jitter ----
    if task_id == "task_hard" and not done_1:
        obs_step2 = step_data.get("observation", {})

        current_stage    = obs_step2.get("current_stage", 12)
        stages_remaining = obs_step2.get("stages_remaining", N_STAGES - current_stage)

        user_msg_2 = build_user_message(obs_step2, task_id, stage=current_stage)
        jitter_note = (
            f"\n[ADVERSARIAL JITTER APPLIED — RECOVERY STEP]\n"
            f"Jitter was injected at global stage {current_stage}. You now have {stages_remaining} stages remaining.\n"
            f"IMPORTANT: In your JSON response, number start_stage and end_stage from 1 to {stages_remaining} "
            f"(NOT 1-24 — use relative stage numbers for the recovery plan).\n"
            f"Example: start_stage=1, end_stage={stages_remaining} means move throughout all remaining stages.\n"
            f"The tooth table below shows the CURRENT (jittered) position as 'current' and the target as 'target'.\n"
            f"Focus on recovery: bring ALL perturbed teeth smoothly back toward target within {stages_remaining} stages.\n\n"
        )
        user_msg_2 = jitter_note + user_msg_2

        llm_result_2  = call_battisibot(client, user_msg_2)
        tooth_plans_2 = llm_result_2["tooth_plans"]
        plan_map_2    = {p["tooth_id"]: p for p in tooth_plans_2}

        initial_poses_2, target_poses_2 = _extract_initial_target_poses(obs_step2)

        all_tooth_trajectories_2: List[List[List[float]]] = []
        for i, tid in enumerate(TOOTH_IDS):
            plan = plan_map_2.get(tid, {
                "start_stage": 1,
                "end_stage":   stages_remaining,
                "ease_in":     0.3,
                "ease_out":    0.3,
            })
            traj = compute_tooth_trajectory(
                initial_pose=initial_poses_2[i],
                target_pose=target_poses_2[i],
                start_stage=plan["start_stage"],
                end_stage=min(plan["end_stage"], stages_remaining),
                ease_in=plan["ease_in"],
                ease_out=plan["ease_out"],
                n_stages=stages_remaining,
            )
            all_tooth_trajectories_2.append(traj)

        trajectory_stages_2: List[Dict] = []
        for s in range(stages_remaining):
            stage_poses = [all_tooth_trajectories_2[i][s] for i in range(N_TEETH)]
            trajectory_stages_2.append({
                "stage_index": s + 1,
                "tooth_ids":   TOOTH_IDS,
                "poses":       stage_poses,
            })

        trajectory_stages_2 = enforce_clinical_constraints(
            trajectory_stages_2, initial_poses_2, tooth_plans_2
        )

        action_payload_2 = {
            "action": {
                "trajectory": trajectory_stages_2,
                "reasoning":  llm_result_2["reasoning"],
                "confidence": llm_result_2["confidence"],
                "metadata":   {},
            }
        }

        step_data_2 = _http_post(f"{space_url}/step", action_payload_2, timeout=120)

        n_steps += 1
        reward_2 = step_data_2.get("reward")
        done_2   = step_data_2.get("done", True)
        if reward_2 is None:
            reward_2 = 0.0

        total_reward = float(reward_2)
        step_rewards.append(total_reward)
        print(
            f"[STEP] step={n_steps} action='recovery plan for {stages_remaining} remaining stages' "
            f"reward={total_reward:.2f} done={str(done_2).lower()} error=null",
            flush=True,
        )

    success     = total_reward >= 0.5
    rewards_str = ",".join(f"{r:.2f}" for r in step_rewards)
    print(
        f"[END] success={str(success).lower()} steps={n_steps} "
        f"score={total_reward:.2f} rewards={rewards_str}",
        flush=True,
    )
    return total_reward, n_steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all three evaluation tasks and print a summary."""
    t_start = time.time()

    client = OpenAI(
        api_key=HF_TOKEN if HF_TOKEN else "dummy",
        base_url=API_BASE_URL,
    )

    tasks = ["task_easy", "task_medium", "task_hard"]
    scores: Dict[str, float] = {}

    for task_id in tasks:
        try:
            reward, steps = run_task(client, task_id, HF_SPACE_URL)
        except Exception as exc:
            print(f"[ERROR] task={task_id} exception={exc}", flush=True)
            reward = 0.0
            print(f"[END] success=false steps=0 score=0.00 rewards=0.00", flush=True)

        scores[task_id] = reward

        elapsed = time.time() - t_start
        if elapsed > 1080:
            print(
                f"[WARN] Elapsed {elapsed:.0f}s exceeds 18-minute soft limit. "
                "Remaining tasks may time out."
            )

    avg_reward = sum(scores.values()) / len(scores)
    print("\n=== battisiBot Evaluation Summary ===")
    print(f"Scores  : {scores}")
    print(f"Average : {avg_reward:.4f}")
    print(f"Elapsed : {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
