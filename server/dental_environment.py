"""
Dental Aligner Trajectory Planning Environment.

Episode structure:
  Step 1: Agent receives initial+target config, submits 24-stage plan.
  [task_hard] Step 2: Adversarial jitter applied, agent revises remaining stages.
  Episode ends after 3 steps max, or after step 1 for easy/medium.
"""
import json
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from openenv.core.env_server.interfaces import Environment

from models import AlignerAction, AlignerObservation, AlignerState, ToothPoseTableRow
from server.synthetic_data import DentalCaseGenerator
from server.grader import AlignerGrader
from server.dental_constants import (
    TOOTH_IDS, TOOTH_TYPES, N_TEETH, N_STAGES,
    ARCH_ADJACENCY, STAGING_PRIORITY,
)
from server.quaternion_utils import quaternion_to_angle_deg, quaternion_multiply, quaternion_inverse


# ---------------------------------------------------------------------------
# Task description strings
# ---------------------------------------------------------------------------

TASK_DESCRIPTION_EASY = """DENTAL ALIGNER PLANNING TASK (EASY) — battisiBot
You are an expert orthodontic treatment planning AI.
Given 28 teeth in malocclusion, plan exactly 24 aligner stages that smoothly
move each tooth from its current pose to its target pose.

Each tooth is a 7-vector: [qw, qx, qy, qz, tx, ty, tz]
  qw,qx,qy,qz = unit quaternion rotation (must satisfy qw^2+qx^2+qy^2+qz^2=1)
  tx, ty, tz  = translation in millimetres

CLINICAL CONSTRAINTS (enforced by grader):
  - Max translation per tooth per stage: 0.25 mm
  - Max rotation per tooth per stage: 2.0 degrees
  - All quaternions must be unit quaternions
  - Stage 0=initial (given), Stage 25=target (given). Output stages 1-24.

SCORING: final_accuracy 40% + smoothness 20% + compliance 20% + staging_quality 20%
A naive SLERP baseline scores ~0.40. Beat it by prioritising incisors first.

HINT: Use SLERP interpolation as your baseline. Move incisors (teeth 11-13,21-23,41-43,31-33) earlier, molars (16,17,26,27,36,37,46,47) later.""".strip()

TASK_DESCRIPTION_MEDIUM = TASK_DESCRIPTION_EASY.replace(
    'DENTAL ALIGNER PLANNING TASK (EASY)',
    'DENTAL ALIGNER PLANNING TASK (MEDIUM)',
)

TASK_DESCRIPTION_HARD = (
    TASK_DESCRIPTION_EASY.replace(
        'DENTAL ALIGNER PLANNING TASK (EASY)',
        'DENTAL ALIGNER PLANNING TASK (HARD)',
    )
    + '\n\nADVERSARIAL MODE: After your initial plan, jitter will be injected into one stage\n'
    '(simulating patient non-compliance). You will then revise remaining stages.\n'
    'Recovery quality contributes 15% to your final score.'
)

_TASK_DESCRIPTIONS = {
    'easy':   TASK_DESCRIPTION_EASY,
    'medium': TASK_DESCRIPTION_MEDIUM,
    'hard':   TASK_DESCRIPTION_HARD,
}


# ---------------------------------------------------------------------------
# Module-level session store so state persists across HTTP request cycles
# ---------------------------------------------------------------------------
_SESSIONS: Dict[str, dict] = {}
_LAST_EPISODE_ID: Optional[str] = None


class DentalAlignerEnvironment(Environment):
    """
    OpenEnv environment for dental aligner trajectory planning.

    SUPPORTS_CONCURRENT_SESSIONS = True: session state is stored in
    module-level dict keyed by episode_id.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True
    MAX_STEPS = 3

    def __init__(self):
        super().__init__()
        self._case_gen = DentalCaseGenerator()
        self._grader = AlignerGrader()

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AlignerObservation:
        """
        Reset — start a new episode.

        Args:
            seed:       Random seed. Derived from episode_id hash if not provided.
            episode_id: Episode ID. Auto-generated UUID if not provided.
            task_id:    'task_easy', 'task_medium', or 'task_hard'.
                        Defaults to 'task_easy' if not provided.
        """
        global _LAST_EPISODE_ID

        # 1. Generate episode_id if not provided
        if episode_id is None:
            episode_id = str(uuid.uuid4())

        # 2. Set seed from episode_id hash if not provided
        if seed is None:
            seed = abs(hash(episode_id)) % (2 ** 31)

        # 3. Determine difficulty from task_id
        difficulty_map = {
            'task_easy':   'easy',
            'task_medium': 'medium',
            'task_hard':   'hard',
        }
        if task_id is None:
            task_id = 'task_easy'
        difficulty = difficulty_map.get(task_id, 'easy')

        # 4. Generate case
        case = self._case_gen.generate_case(difficulty, seed)

        # 5. Store session
        _SESSIONS[episode_id] = {
            'task_id':             task_id,
            'difficulty':          difficulty,
            'case':                case,
            'step':                0,
            'pre_jitter_accuracy': 0.0,
            'adv_stages_used':     0,
            'last_agent_traj':     None,
            'seed':                seed,
        }
        _LAST_EPISODE_ID = episode_id

        # 6. Build initial observation
        initial_config = case['initial_config']
        target_config  = case['target_config']
        baseline_traj  = case['baseline_trajectory']

        tooth_table      = self._build_tooth_table(initial_config, target_config)
        tooth_table_text = self._build_tooth_table_text(tooth_table)
        arch_graph_json  = self._build_arch_graph_json()
        baseline_json    = self._build_baseline_json(baseline_traj)

        task_desc = _TASK_DESCRIPTIONS[difficulty]

        return AlignerObservation(
            done=False,
            reward=None,
            task_id=task_id,
            current_stage=0,
            stages_remaining=N_STAGES,
            task_description=task_desc,
            tooth_table=tooth_table,
            tooth_table_text=tooth_table_text,
            arch_graph_json=arch_graph_json,
            baseline_trajectory_json=baseline_json,
            last_plan_feedback='',
            jitter_description='',
            step_number=0,
            adversarial_jitter_applied=False,
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self,
        action: AlignerAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> AlignerObservation:
        """
        Grade the agent's trajectory plan and return feedback.

        Step 1 (all difficulties): Grade submitted 24-stage plan.
        Step 2 (task_hard only):   Apply adversarial jitter and ask agent to revise.
        Episode done after step 1 (easy/medium) or step 2 (hard).
        """
        episode_id = _LAST_EPISODE_ID
        if episode_id is None or episode_id not in _SESSIONS:
            raise RuntimeError('No active session. Call /reset before /step.')

        session    = _SESSIONS[episode_id]
        case       = session['case']
        task_id    = session['task_id']
        difficulty = session['difficulty']
        step_num   = session['step'] + 1
        session['step'] = step_num

        initial_config = case['initial_config']   # (28, 7)
        target_config  = case['target_config']    # (28, 7)
        baseline_traj  = case['baseline_trajectory']  # (26, 28, 7)

        # --- Determine context depending on hard step 2 ---
        is_hard_step2 = (difficulty == 'hard' and step_num == 2)

        if is_hard_step2:
            # Step 2: agent revises from the jitter point onward
            current_traj = session.get('jittered_traj', baseline_traj)
            jitter_stage = session.get('jitter_stage', 12)
            stages_remaining = N_STAGES - jitter_stage
        else:
            # Step 1: full 24-stage plan
            current_traj = baseline_traj
            jitter_stage = None
            stages_remaining = N_STAGES

        # --- Parse agent's trajectory ---
        agent_traj = self._parse_agent_trajectory(
            action, initial_config, target_config, stages_remaining
        )

        # For hard step 2, splice revised stages back into the current trajectory
        if is_hard_step2 and jitter_stage is not None:
            full_traj = current_traj.copy()
            revised_count = agent_traj.shape[0] - 2  # excludes stage 0 and 25 padding
            for s in range(revised_count):
                global_stage = jitter_stage + 1 + s
                if global_stage < 25:
                    full_traj[global_stage] = agent_traj[1 + s]
            agent_traj_for_grade = full_traj
        else:
            agent_traj_for_grade = agent_traj

        # --- Grade ---
        reward, feedback = self._grader.grade(
            task_id=task_id,
            agent_traj=agent_traj_for_grade,
            initial=initial_config,
            target=target_config,
            adv_stages=session['adv_stages_used'],
            pre_jitter_accuracy=session['pre_jitter_accuracy'],
        )

        # Store agent trajectory in session
        session['last_agent_traj'] = agent_traj_for_grade

        # --- task_hard step 1: apply adversarial jitter ---
        adv_jitter_applied  = False
        jittered_stage_out  = None
        jittered_teeth_out  = None

        if difficulty == 'hard' and step_num == 1:
            # Record pre-jitter accuracy for step 2 grading
            session['pre_jitter_accuracy'] = reward

            # Apply jitter to a mid-point stage
            jitter_stage = 12
            rng = np.random.default_rng(session['seed'] + 1)
            jitter_strength = 0.2
            jittered_traj, jittered_teeth = self._case_gen.apply_adversarial_jitter(
                agent_traj_for_grade, jitter_stage, jitter_strength, rng
            )
            session['jittered_traj']   = jittered_traj
            session['jitter_stage']    = jitter_stage
            session['jittered_teeth']  = jittered_teeth
            session['adv_stages_used'] = 1   # jitter applied; enables recovery bonus
            adv_jitter_applied  = True
            jittered_stage_out  = jitter_stage
            jittered_teeth_out  = jittered_teeth

            # Build revised obs (done=False for hard, agent must submit step 2)
            current_config = jittered_traj[jitter_stage]
            tooth_table      = self._build_tooth_table(current_config, target_config)
            tooth_table_text = self._build_tooth_table_text(tooth_table)
            arch_graph_json  = self._build_arch_graph_json()
            baseline_json    = self._build_baseline_json(baseline_traj)

            return AlignerObservation(
                done=False,
                reward=None,   # no reward yet — episode continues
                task_id=task_id,
                current_stage=jitter_stage,
                stages_remaining=N_STAGES - jitter_stage,
                task_description=_TASK_DESCRIPTIONS[difficulty],
                tooth_table=tooth_table,
                tooth_table_text=tooth_table_text,
                arch_graph_json=arch_graph_json,
                baseline_trajectory_json=baseline_json,
                last_plan_feedback=feedback,
                jitter_description=(
                    f'Stage {jitter_stage} was perturbed on teeth: {jittered_teeth}. '
                    f'Revise stages {jitter_stage+1}-24.'
                ),
                step_number=step_num,
                adversarial_jitter_applied=True,
            )

        # --- Determine done ---
        done = True  # easy and medium always done after step 1
        if difficulty == 'hard' and step_num < 2:
            done = False
        if step_num >= self.MAX_STEPS:
            done = True

        # --- Build final observation ---
        tooth_table      = self._build_tooth_table(initial_config, target_config)
        tooth_table_text = self._build_tooth_table_text(tooth_table)
        arch_graph_json  = self._build_arch_graph_json()
        baseline_json    = self._build_baseline_json(baseline_traj)

        return AlignerObservation(
            done=done,
            reward=reward,
            task_id=task_id,
            current_stage=N_STAGES if not is_hard_step2 else (jitter_stage or N_STAGES),
            stages_remaining=0,
            task_description=_TASK_DESCRIPTIONS[difficulty],
            tooth_table=tooth_table,
            tooth_table_text=tooth_table_text,
            arch_graph_json=arch_graph_json,
            baseline_trajectory_json=baseline_json,
            last_plan_feedback=feedback,
            jitter_description='',
            step_number=step_num,
            adversarial_jitter_applied=adv_jitter_applied,
        )

    # ------------------------------------------------------------------
    # state property
    # ------------------------------------------------------------------

    @property
    def state(self) -> AlignerState:
        """Current episode state."""
        if _LAST_EPISODE_ID and _LAST_EPISODE_ID in _SESSIONS:
            session = _SESSIONS[_LAST_EPISODE_ID]
            return AlignerState(
                episode_id=_LAST_EPISODE_ID,
                step_count=session['step'],
                task_id=session['task_id'],
                difficulty=session['difficulty'],
                seed=session['seed'],
                current_stage=min(session['step'] * N_STAGES, N_STAGES),
                total_violations=0,
                adversarial_perturbations=session['adv_stages_used'],
                best_trajectory_score=session['pre_jitter_accuracy'],
            )
        return AlignerState()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_tooth_table(
        self,
        current_config: np.ndarray,
        target_config: np.ndarray,
    ) -> List[ToothPoseTableRow]:
        """Build list of 28 ToothPoseTableRow entries."""
        rows = []
        for i, tooth_id in enumerate(TOOTH_IDS):
            curr = current_config[i]   # (7,)
            tgt  = target_config[i]    # (7,)

            # Translation distance (mm)
            dist_mm = float(np.linalg.norm(curr[4:7] - tgt[4:7]))

            # Rotation distance (degrees): angle of relative rotation
            q_curr = curr[:4]
            q_tgt  = tgt[:4]
            q_inv  = quaternion_inverse(q_curr)
            q_rel  = quaternion_multiply(q_tgt, q_inv)
            dist_deg = quaternion_to_angle_deg(q_rel)

            rows.append(ToothPoseTableRow(
                tooth_id=tooth_id,
                tooth_type=TOOTH_TYPES[tooth_id],
                current_qw=float(curr[0]), current_qx=float(curr[1]),
                current_qy=float(curr[2]), current_qz=float(curr[3]),
                current_tx=float(curr[4]), current_ty=float(curr[5]),
                current_tz=float(curr[6]),
                target_qw=float(tgt[0]), target_qx=float(tgt[1]),
                target_qy=float(tgt[2]), target_qz=float(tgt[3]),
                target_tx=float(tgt[4]), target_ty=float(tgt[5]),
                target_tz=float(tgt[6]),
                remaining_trans_mm=round(dist_mm, 3),
                remaining_rot_deg=round(dist_deg, 3),
            ))
        return rows

    def _build_tooth_table_text(
        self, tooth_table: List[ToothPoseTableRow]
    ) -> str:
        """Build a markdown table for human-readable display."""
        header = (
            '| Tooth | Type              | CurrPos(mm)            '
            '| TargetPos(mm)          | Dist_mm | Dist_deg |\n'
            '|-------|-------------------|------------------------|'
            '------------------------|---------|----------|\n'
        )
        lines = [header]
        for row in tooth_table:
            curr_str   = f'({row.current_tx:.1f}, {row.current_ty:.1f}, {row.current_tz:.1f})'
            target_str = f'({row.target_tx:.1f}, {row.target_ty:.1f}, {row.target_tz:.1f})'
            lines.append(
                f'| {row.tooth_id:5d} | {row.tooth_type:17s} | {curr_str:22s} '
                f'| {target_str:22s} | {row.remaining_trans_mm:7.3f} | {row.remaining_rot_deg:8.3f} |\n'
            )
        return ''.join(lines)

    def _build_arch_graph_json(self) -> str:
        """Build adjacency list from ARCH_ADJACENCY, serialised to JSON."""
        adjacency: Dict[int, List[int]] = {}
        for (a, b) in ARCH_ADJACENCY:
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
        # JSON keys must be strings
        return json.dumps({str(k): v for k, v in sorted(adjacency.items())})

    def _build_baseline_json(self, baseline_trajectory: np.ndarray) -> str:
        """Serialise stages 1-24 of the SLERP baseline to compact JSON."""
        # baseline_trajectory shape: (26, 28, 7)
        stages = {}
        for s in range(1, 25):
            stages[str(s)] = baseline_trajectory[s].tolist()
        return json.dumps(stages, separators=(',', ':'))

    def _parse_agent_trajectory(
        self,
        action: AlignerAction,
        initial_config: np.ndarray,
        target_config: np.ndarray,
        stages_remaining: int,
    ) -> np.ndarray:
        """
        Parse action.trajectory into a numpy array of shape (26, 28, 7).

        - action.trajectory is expected to be a dict keyed by stage number (int or str)
          mapping to a list of 28 poses each of length 7, OR a list of 24 (or fewer)
          stages where each stage is a list of 28 x 7 vectors.
        - Quaternions are normalised.
        - Padded: stage 0 = initial_config, stage 25 = target_config.
        """
        traj = np.zeros((26, N_TEETH, 7), dtype=np.float64)
        traj[0]  = initial_config.copy()
        traj[25] = target_config.copy()

        # action.trajectory is List[ToothTrajectoryStage]
        # Each stage has .stage_index (1-24) and .poses (list of 28 x 7-float lists)

        def _fill_stage(stage_idx: int, pose_list: Any) -> None:
            """Fill traj[stage_idx] from a list-of-28-pose-7-vectors."""
            if stage_idx < 1 or stage_idx > 24:
                return
            if not pose_list or len(pose_list) == 0:
                return
            arr = np.array(pose_list, dtype=np.float64)
            if arr.shape != (N_TEETH, 7):
                return
            # Normalise quaternions
            for i in range(N_TEETH):
                q = arr[i, :4]
                n = np.linalg.norm(q)
                if n > 1e-10:
                    arr[i, :4] = q / n
                else:
                    arr[i, :4] = initial_config[i, :4]
            traj[stage_idx] = arr

        raw = action.trajectory
        if raw:
            for stage_obj in raw:
                # stage_obj is a ToothTrajectoryStage (Pydantic model)
                if hasattr(stage_obj, 'stage_index') and hasattr(stage_obj, 'poses'):
                    _fill_stage(int(stage_obj.stage_index), stage_obj.poses)
                elif isinstance(stage_obj, dict):
                    _fill_stage(int(stage_obj.get('stage_index', 0)), stage_obj.get('poses', []))

        # Fill any zero (unpopulated) stages with SLERP interpolation
        for s in range(1, 25):
            if np.allclose(traj[s], 0.0):
                alpha = s / 25.0
                for i in range(N_TEETH):
                    from server.quaternion_utils import quaternion_slerp, quaternion_normalize
                    traj[s, i, :4] = quaternion_normalize(
                        quaternion_slerp(traj[0, i, :4], traj[25, i, :4], alpha)
                    )
                    traj[s, i, 4:7] = (1.0 - alpha) * traj[0, i, 4:7] + alpha * traj[25, i, 4:7]

        return traj
