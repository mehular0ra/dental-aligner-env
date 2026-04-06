"""
Grader for the Dental Aligner Trajectory Planning environment.

Scoring design:
  - Multi-component rewards with weighted sums per difficulty.
  - Biomechanical constraint compliance penalised directly (not binary).
  - Staging quality measured via Spearman rank correlation with clinical priority.
  - Adversarial recovery bonus rewards robustness to jitter on hard tasks.
"""

import math
import numpy as np
from scipy.stats import spearmanr
from typing import Dict, List, Tuple, Any

from .dental_constants import (
    N_TEETH, N_STAGES, TOOTH_IDS, TOOTH_TYPES, STAGING_PRIORITY,
    MAX_TRANSLATION_PER_STAGE_MM, MAX_ROTATION_PER_STAGE_DEG,
)
from .quaternion_utils import quaternion_multiply, quaternion_inverse, quaternion_to_angle_deg


def _ensure_full_trajectory(agent_trajectory: np.ndarray) -> np.ndarray:
    """
    Accept trajectories of shape (26, 28, 7) or shorter and return as-is.
    Validates that last dimension is 7 (quaternion + translation).
    Raises ValueError for completely invalid shapes.
    """
    traj = np.asarray(agent_trajectory, dtype=np.float64)
    if traj.ndim != 3 or traj.shape[1] != N_TEETH or traj.shape[2] != 7:
        raise ValueError(
            f"agent_trajectory must have shape (n_stages, {N_TEETH}, 7), "
            f"got {traj.shape}"
        )
    return traj


class AlignerGrader:
    """Grades agent-planned aligner trajectories against target configurations."""

    # ------------------------------------------------------------------
    # Component scorers
    # ------------------------------------------------------------------

    def compute_final_accuracy(
        self,
        agent_trajectory: np.ndarray,
        target_config: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Measure how closely the last stage of the trajectory matches the target.

        Parameters
        ----------
        agent_trajectory : np.ndarray, shape (n_stages, 28, 7)
            Full padded trajectory; stage 0 = initial, stage -1 = final agent output.
        target_config : np.ndarray, shape (28, 7)
            Ground-truth target pose for every tooth.

        Returns
        -------
        dict with keys:
            final_accuracy  — float in [0.0, 1.0], mean over all teeth
            per_tooth       — List[float] of per-tooth scores (length 28)
            errors_mm       — List[float] translational errors in mm
            errors_deg      — List[float] rotational errors in degrees
        """
        traj = _ensure_full_trajectory(agent_trajectory)
        target = np.asarray(target_config, dtype=np.float64)

        # Use stage 24 (last agent-provided stage), NOT stage 25 (the target itself).
        # Trajectory: stage 0=initial, stages 1-24=agent plan, stage 25=target.
        # traj[-1]=target would trivially score 1.0; we evaluate the agent's stage 24.
        final_stage_idx = min(24, traj.shape[0] - 2)
        final_stage = traj[final_stage_idx]  # shape (28, 7)

        per_tooth: List[float] = []
        errors_mm: List[float] = []
        errors_deg: List[float] = []

        for i in range(N_TEETH):
            # Translation error (mm)
            trans_error = float(np.linalg.norm(final_stage[i, 4:7] - target[i, 4:7]))

            # Rotation error (degrees) — q_rel = q_agent * q_target^{-1}
            q_rel = quaternion_multiply(
                final_stage[i, :4],
                quaternion_inverse(target[i, :4]),
            )
            rot_error = quaternion_to_angle_deg(q_rel)

            trans_score = max(0.0, 1.0 - trans_error / 2.0)
            rot_score = max(0.0, 1.0 - rot_error / 10.0)
            tooth_score = 0.6 * trans_score + 0.4 * rot_score

            per_tooth.append(tooth_score)
            errors_mm.append(trans_error)
            errors_deg.append(rot_error)

        final_accuracy = float(np.mean(per_tooth))

        return {
            'final_accuracy': final_accuracy,
            'per_tooth': per_tooth,
            'errors_mm': errors_mm,
            'errors_deg': errors_deg,
        }

    def compute_smoothness(self, agent_trajectory: np.ndarray) -> float:
        """
        Measure movement smoothness across stages.

        For each consecutive stage pair (k, k+1) and each tooth i, compute the
        translational step size delta_t[k][i].  Smoothness penalises variance
        in these step sizes: highly uneven movements (large bursts followed by
        stalls) score low.

        Returns float in [0.0, 1.0].
        """
        traj = _ensure_full_trajectory(agent_trajectory)
        n_stages = traj.shape[0]

        if n_stages < 2:
            return 1.0  # degenerate — treat as perfectly smooth

        all_deltas: List[float] = []

        for k in range(n_stages - 1):
            for i in range(N_TEETH):
                delta = float(np.linalg.norm(traj[k + 1, i, 4:7] - traj[k, i, 4:7]))
                all_deltas.append(delta)

        if not all_deltas:
            return 1.0

        variance = float(np.var(all_deltas))
        smoothness = 1.0 - min(1.0, variance / 0.05)
        return max(0.0, smoothness)

    def compute_constraint_compliance(
        self, agent_trajectory: np.ndarray
    ) -> Dict[str, Any]:
        """
        Check per-stage biomechanical movement limits.

        For each consecutive stage pair (k, k+1) and each tooth i, a violation
        occurs when the translational or rotational delta exceeds the clinical
        per-stage limits defined in dental_constants.

        Returns
        -------
        dict with keys:
            compliance_score  — float in [0.0, 1.0]
            n_violations      — int total violation count
            violation_details — List[dict] with stage, tooth_index, tooth_id,
                                trans_delta, rot_delta, type
        """
        traj = _ensure_full_trajectory(agent_trajectory)
        n_stages = traj.shape[0]

        n_violations = 0
        violation_details: List[Dict[str, Any]] = []

        # Only check stages 0→1 through 23→24 (agent-driven transitions).
        # Stage 24→25 is not an aligner — 25 is the ideal target reference.
        n_check = min(N_STAGES, n_stages - 1)  # 24
        for k in range(n_check):
            for i in range(N_TEETH):
                # Translational delta
                trans_delta = float(
                    np.linalg.norm(traj[k + 1, i, 4:7] - traj[k, i, 4:7])
                )

                # Rotational delta
                q_rel = quaternion_multiply(
                    traj[k + 1, i, :4],
                    quaternion_inverse(traj[k, i, :4]),
                )
                rot_delta = quaternion_to_angle_deg(q_rel)

                trans_violated = trans_delta > MAX_TRANSLATION_PER_STAGE_MM
                rot_violated = rot_delta > MAX_ROTATION_PER_STAGE_DEG

                if trans_violated or rot_violated:
                    n_violations += 1
                    violation_type = []
                    if trans_violated:
                        violation_type.append('translation')
                    if rot_violated:
                        violation_type.append('rotation')
                    violation_details.append({
                        'stage': k,
                        'tooth_index': i,
                        'tooth_id': TOOTH_IDS[i],
                        'trans_delta_mm': round(trans_delta, 4),
                        'rot_delta_deg': round(rot_delta, 4),
                        'type': '+'.join(violation_type),
                    })

        max_possible_violations = N_TEETH * n_check  # N_TEETH * N_STAGES = 28 * 24 = 672
        if max_possible_violations == 0:
            compliance_score = 1.0
        else:
            compliance_score = 1.0 - (n_violations / max_possible_violations)

        return {
            'compliance_score': max(0.0, compliance_score),
            'n_violations': n_violations,
            'violation_details': violation_details,
        }

    def compute_staging_quality(
        self,
        agent_trajectory: np.ndarray,
        initial_config: np.ndarray,
        target_config: np.ndarray,
    ) -> float:
        """
        Measure whether teeth move in the clinically correct priority order.

        For each tooth, find the first stage at which the cumulative
        translational displacement from the initial position exceeds 0.1 mm.
        Compare the induced ordering of teeth by movement_start_stage with the
        clinical STAGING_PRIORITY order using Spearman rank correlation.

        Returns float in [0.0, 1.0].
        """
        traj = _ensure_full_trajectory(agent_trajectory)
        initial = np.asarray(initial_config, dtype=np.float64)

        n_stages = traj.shape[0]

        # movement_start_stage[i] = first stage index where cumulative
        # translation from initial > 0.1 mm, or n_stages if never reached.
        movement_start: List[int] = []
        for i in range(N_TEETH):
            started = n_stages  # default: tooth never moves
            for k in range(1, n_stages):
                cum_trans = float(np.linalg.norm(traj[k, i, 4:7] - initial[i, 4:7]))
                if cum_trans > 0.1:
                    started = k
                    break
            movement_start.append(started)

        # Edge case: all teeth start moving at the same stage
        if len(set(movement_start)) == 1:
            return 0.5

        # Map each tooth's type to its STAGING_PRIORITY rank (lower = earlier)
        priority_ranks: List[int] = []
        for i in range(N_TEETH):
            tooth_type = TOOTH_TYPES[TOOTH_IDS[i]]
            try:
                rank = STAGING_PRIORITY.index(tooth_type)
            except ValueError:
                rank = len(STAGING_PRIORITY)  # unknown type → lowest priority
            priority_ranks.append(rank)

        # Spearman correlation: good staging = teeth with low priority_rank start early
        result = spearmanr(priority_ranks, movement_start)
        rho = float(result.correlation)

        # Handle NaN (e.g., all values identical after edge case above)
        if math.isnan(rho):
            return 0.5

        # Normalise from [-1, 1] to [0, 1]
        return (rho + 1.0) / 2.0

    # ------------------------------------------------------------------
    # Task-level graders
    # ------------------------------------------------------------------

    def _worst_teeth_feedback(
        self,
        per_tooth_scores: List[float],
        n: int = 3,
    ) -> str:
        """Return a human-readable string identifying the n worst-scoring teeth."""
        indexed = sorted(enumerate(per_tooth_scores), key=lambda x: x[1])
        lines = []
        for rank, (i, score) in enumerate(indexed[:n], start=1):
            tid = TOOTH_IDS[i]
            ttype = TOOTH_TYPES[tid]
            lines.append(
                f"  {rank}. Tooth {tid} ({ttype}): accuracy score = {score:.3f}"
            )
        return "Worst teeth by final accuracy:\n" + "\n".join(lines)

    def grade_easy(
        self,
        agent_traj: np.ndarray,
        initial: np.ndarray,
        target: np.ndarray,
    ) -> Tuple[float, str]:
        """
        Easy task grader.

        Weights: final_accuracy=0.40, smoothness=0.20, compliance=0.20,
                 staging_quality=0.20

        Returns (reward, feedback_str).
        """
        accuracy_result = self.compute_final_accuracy(agent_traj, target)
        smoothness = self.compute_smoothness(agent_traj)
        compliance_result = self.compute_constraint_compliance(agent_traj)
        staging_quality = self.compute_staging_quality(agent_traj, initial, target)

        final_accuracy = accuracy_result['final_accuracy']
        compliance_score = compliance_result['compliance_score']
        n_violations = compliance_result['n_violations']

        reward = (
            0.40 * final_accuracy
            + 0.20 * smoothness
            + 0.20 * compliance_score
            + 0.20 * staging_quality
        )
        reward = max(0.0, min(1.0, reward))

        worst_feedback = self._worst_teeth_feedback(accuracy_result['per_tooth'], n=3)
        feedback = (
            f"[Easy] reward={reward:.4f}\n"
            f"  Components:\n"
            f"    final_accuracy  = {final_accuracy:.4f}  (w=0.40)\n"
            f"    smoothness      = {smoothness:.4f}  (w=0.20)\n"
            f"    compliance      = {compliance_score:.4f}  (w=0.20, {n_violations} violations)\n"
            f"    staging_quality = {staging_quality:.4f}  (w=0.20)\n"
            + worst_feedback
        )
        return reward, feedback

    def grade_medium(
        self,
        agent_traj: np.ndarray,
        initial: np.ndarray,
        target: np.ndarray,
    ) -> Tuple[float, str]:
        """
        Medium task grader.

        Weights: final_accuracy=0.45, smoothness=0.20, compliance=0.20,
                 staging_quality=0.15

        Returns (reward, feedback_str).
        """
        accuracy_result = self.compute_final_accuracy(agent_traj, target)
        smoothness = self.compute_smoothness(agent_traj)
        compliance_result = self.compute_constraint_compliance(agent_traj)
        staging_quality = self.compute_staging_quality(agent_traj, initial, target)

        final_accuracy = accuracy_result['final_accuracy']
        compliance_score = compliance_result['compliance_score']
        n_violations = compliance_result['n_violations']

        reward = (
            0.45 * final_accuracy
            + 0.20 * smoothness
            + 0.20 * compliance_score
            + 0.15 * staging_quality
        )
        reward = max(0.0, min(1.0, reward))

        worst_feedback = self._worst_teeth_feedback(accuracy_result['per_tooth'], n=5)
        feedback = (
            f"[Medium] reward={reward:.4f}\n"
            f"  Components:\n"
            f"    final_accuracy  = {final_accuracy:.4f}  (w=0.45)\n"
            f"    smoothness      = {smoothness:.4f}  (w=0.20)\n"
            f"    compliance      = {compliance_score:.4f}  (w=0.20, {n_violations} violations)\n"
            f"    staging_quality = {staging_quality:.4f}  (w=0.15)\n"
            + worst_feedback
        )
        return reward, feedback

    def grade_hard(
        self,
        agent_traj: np.ndarray,
        initial: np.ndarray,
        target: np.ndarray,
        adversarial_stages_used: int = 0,
        pre_jitter_accuracy: float = 0.0,
    ) -> Tuple[float, str]:
        """
        Hard task grader.

        Weights: final_accuracy=0.40, smoothness=0.15, compliance=0.15,
                 staging_quality=0.15, recovery_bonus=0.15

        Recovery bonus:
          - 0.0 if adversarial_stages_used == 0 (no jitter applied)
          - Otherwise: max(0, (post_jitter_accuracy - pre_jitter_accuracy) / 0.5) * 0.15

        Hard penalty: if n_violations > 20, final reward *= 0.5

        Returns (reward, feedback_str).
        """
        accuracy_result = self.compute_final_accuracy(agent_traj, target)
        smoothness = self.compute_smoothness(agent_traj)
        compliance_result = self.compute_constraint_compliance(agent_traj)
        staging_quality = self.compute_staging_quality(agent_traj, initial, target)

        final_accuracy = accuracy_result['final_accuracy']
        compliance_score = compliance_result['compliance_score']
        n_violations = compliance_result['n_violations']

        # Recovery bonus
        if adversarial_stages_used == 0:
            recovery_bonus_raw = 0.0
            recovery_bonus_weighted = 0.0
        else:
            post_jitter_accuracy = final_accuracy
            recovery_ratio = max(
                0.0,
                (post_jitter_accuracy - pre_jitter_accuracy) / 0.5,
            )
            recovery_bonus_raw = min(1.0, recovery_ratio)
            recovery_bonus_weighted = recovery_bonus_raw * 0.15

        reward = (
            0.40 * final_accuracy
            + 0.15 * smoothness
            + 0.15 * compliance_score
            + 0.15 * staging_quality
            + recovery_bonus_weighted
        )

        # Hard penalty for excessive biomechanical violations
        hard_penalty_applied = n_violations > 20
        if hard_penalty_applied:
            reward *= 0.5

        reward = max(0.0, min(1.0, reward))

        worst_feedback = self._worst_teeth_feedback(accuracy_result['per_tooth'], n=5)
        penalty_note = ' [x0.5 hard penalty: >20 violations]' if hard_penalty_applied else ''
        feedback = (
            f"[Hard] reward={reward:.4f}{penalty_note}\n"
            f"  Components:\n"
            f"    final_accuracy   = {final_accuracy:.4f}  (w=0.40)\n"
            f"    smoothness       = {smoothness:.4f}  (w=0.15)\n"
            f"    compliance       = {compliance_score:.4f}  (w=0.15, {n_violations} violations)\n"
            f"    staging_quality  = {staging_quality:.4f}  (w=0.15)\n"
            f"    recovery_bonus   = {recovery_bonus_weighted:.4f}  (w=0.15, "
            f"adv_stages={adversarial_stages_used}, pre_acc={pre_jitter_accuracy:.4f})\n"
            + worst_feedback
        )
        return reward, feedback

    def grade(
        self,
        task_id: str,
        agent_traj: np.ndarray,
        initial: np.ndarray,
        target: np.ndarray,
        adv_stages: int = 0,
        pre_jitter_accuracy: float = 0.0,
    ) -> Tuple[float, str]:
        """
        Dispatch to the correct difficulty grader based on task_id.

        Parameters
        ----------
        task_id : str
            One of 'task_easy', 'task_medium', 'task_hard'.
        agent_traj : np.ndarray, shape (n_stages, 28, 7)
            Full padded trajectory (stage 0 = initial, stage -1 = final).
        initial : np.ndarray, shape (28, 7)
            Initial (stage 0) tooth configuration.
        target : np.ndarray, shape (28, 7)
            Ground-truth target tooth configuration.
        adv_stages : int
            Number of adversarial perturbation stages encountered (hard only).
        pre_jitter_accuracy : float
            Final accuracy score measured before jitter was applied (hard only).

        Returns
        -------
        (reward, feedback_str)
        """
        if task_id == 'task_easy':
            return self.grade_easy(agent_traj, initial, target)
        elif task_id == 'task_medium':
            return self.grade_medium(agent_traj, initial, target)
        elif task_id == 'task_hard':
            return self.grade_hard(
                agent_traj, initial, target,
                adversarial_stages_used=adv_stages,
                pre_jitter_accuracy=pre_jitter_accuracy,
            )
        else:
            return 0.0, f'Unknown task_id: {task_id!r}. Expected task_easy, task_medium, or task_hard.'
