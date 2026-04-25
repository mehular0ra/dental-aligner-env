"""
Synthetic dental case generator for the aligner trajectory planner.
Produces deterministic, seeded dental malocclusion cases.
"""
import math
import numpy as np
from typing import Dict, Any

from .dental_constants import (
    TOOTH_IDS, N_TEETH, N_STAGES,
    TOOTH_TYPES, STAGING_PRIORITY,
    IDEAL_UPPER_TX, IDEAL_UPPER_TY, IDEAL_UPPER_TZ,
    IDEAL_LOWER_TX, IDEAL_LOWER_TY, IDEAL_LOWER_TZ,
)
from .quaternion_utils import (
    quaternion_slerp,
    quaternion_multiply,
    quaternion_inverse,
    quaternion_normalize,
    quaternion_from_axis_angle,
    random_quaternion_perturbation,
)
from .clinical_profiles import (
    sample_profile,
    MALOCCLUSION_GEOMETRY,
    CROWDING_PARAMS,
    OVERBITE_PARAMS,
    OVERJET_PARAMS,
)


class DentalCaseGenerator:
    """
    Generates synthetic dental cases with malocclusion perturbations.
    All generation is seeded for reproducibility.
    """

    def generate_ideal_config(self) -> np.ndarray:
        """
        Build the 28x7 ideal dental configuration.
        All rotations are identity quaternion (1,0,0,0).
        """
        config = np.zeros((N_TEETH, 7), dtype=np.float64)
        config[:, 0] = 1.0  # qw = 1 (identity rotation)

        # Upper arch: first 14 teeth (indices 0-13) = tooth IDs 11-17, 21-27
        for i in range(14):
            config[i, 4] = IDEAL_UPPER_TX[i]
            config[i, 5] = IDEAL_UPPER_TY[i]
            config[i, 6] = IDEAL_UPPER_TZ[i]

        # Lower arch: last 14 teeth (indices 14-27) = tooth IDs 31-37, 41-47
        for i in range(14):
            config[14 + i, 4] = IDEAL_LOWER_TX[i]
            config[14 + i, 5] = IDEAL_LOWER_TY[i]
            config[14 + i, 6] = IDEAL_LOWER_TZ[i]

        return config

    def apply_malocclusion(
        self,
        ideal: np.ndarray,
        difficulty: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Perturb the ideal config to generate a malocclusion case.

        difficulty:
          'easy'   : 4-6 teeth, 1-3mm translation, 5-15 deg rotation (z-axis tipping)
          'medium' : 10-14 teeth, 2-5mm, 10-20 deg (multi-axis)
          'hard'   : 18-24 teeth, 3-8mm, 15-25 deg (combined motions)
        """
        config = ideal.copy()

        if difficulty == 'easy':
            n_perturb = int(rng.integers(4, 7))
            trans_range = (1.0, 3.0)
            rot_range = (5.0, 15.0)
            axes = ['z']
        elif difficulty == 'medium':
            n_perturb = int(rng.integers(10, 15))
            trans_range = (2.0, 5.0)
            rot_range = (10.0, 20.0)
            axes = ['x', 'y', 'z']
        else:  # hard
            n_perturb = int(rng.integers(18, 25))
            trans_range = (3.0, 8.0)
            rot_range = (15.0, 25.0)
            axes = ['x', 'y', 'z']

        indices = rng.choice(N_TEETH, size=n_perturb, replace=False)

        for idx in indices:
            # Apply random translation
            trans_mag = rng.uniform(*trans_range)
            direction = rng.standard_normal(3)
            direction /= (np.linalg.norm(direction) + 1e-12)
            if difficulty == 'easy':
                direction[2] = 0.0  # mostly in-plane for easy
                direction /= (np.linalg.norm(direction) + 1e-12)
            config[idx, 4:7] += direction * trans_mag

            # Apply random rotation
            rot_deg = rng.uniform(*rot_range)
            if 'z' in axes and difficulty == 'easy':
                axis = np.array([0.0, 0.0, 1.0])
            else:
                axis = rng.standard_normal(3)
                axis /= (np.linalg.norm(axis) + 1e-12)

            delta_q = quaternion_from_axis_angle(axis, math.radians(rot_deg))
            old_q = config[idx, :4]
            new_q = quaternion_normalize(quaternion_multiply(delta_q, old_q))
            config[idx, :4] = new_q

        return config

    def generate_baseline_trajectory(
        self,
        initial: np.ndarray,
        final: np.ndarray,
    ) -> np.ndarray:
        """
        Generate a 26x28x7 SLERP baseline trajectory.
        stage 0 = initial, stage 25 = final, stages 1-24 = interpolated.
        """
        trajectory = np.zeros((26, N_TEETH, 7), dtype=np.float64)
        trajectory[0] = initial.copy()
        trajectory[25] = final.copy()

        for k in range(1, 25):
            alpha = k / 25.0
            for i in range(N_TEETH):
                # SLERP rotation
                q0 = initial[i, :4]
                q1 = final[i, :4]
                trajectory[k, i, :4] = quaternion_slerp(q0, q1, alpha)

                # Linear interpolation for translation
                t0 = initial[i, 4:7]
                t1 = final[i, 4:7]
                trajectory[k, i, 4:7] = (1.0 - alpha) * t0 + alpha * t1

        return trajectory

    def compute_delta_poses(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Given 26x28x7 trajectory, compute 25x28x7 delta poses between consecutive stages.
        delta[k][i] = relative pose from stage k to k+1 for tooth i.
        """
        n_steps = trajectory.shape[0] - 1
        deltas = np.zeros((n_steps, N_TEETH, 7), dtype=np.float64)

        for k in range(n_steps):
            for i in range(N_TEETH):
                # Delta translation
                deltas[k, i, 4:7] = trajectory[k+1, i, 4:7] - trajectory[k, i, 4:7]

                # Delta rotation: q_delta = q_{k+1} * q_k^{-1}
                q_k = trajectory[k, i, :4]
                q_k1 = trajectory[k+1, i, :4]
                q_delta = quaternion_normalize(
                    quaternion_multiply(q_k1, quaternion_inverse(q_k))
                )
                deltas[k, i, :4] = q_delta

        return deltas

    def generate_case(self, difficulty: str, seed: int) -> Dict[str, Any]:
        """
        Generate a complete dental case dict.
        Same seed -> same case (reproducibility requirement).
        """
        rng = np.random.default_rng(seed)
        ideal = self.generate_ideal_config()
        initial = self.apply_malocclusion(ideal, difficulty, rng)
        baseline_traj = self.generate_baseline_trajectory(initial, ideal)

        return {
            'initial_config':      initial,        # shape (28, 7)
            'target_config':       ideal,           # shape (28, 7)
            'tooth_ids':           TOOTH_IDS,       # list of 28 FDI IDs
            'tooth_types':         TOOTH_TYPES,
            'baseline_trajectory': baseline_traj,  # shape (26, 28, 7)
            'difficulty':          difficulty,
            'seed':                seed,
        }

    def apply_clinical_perturbation(
        self,
        ideal: np.ndarray,
        profile: Dict[str, Any],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Perturb the ideal config based on a real clinical profile.
        Applies diagnosis-specific geometry shifts before random perturbation.
        """
        config = ideal.copy()
        malocclusion = profile.get("malocclusion", "ClassI")
        crowding = profile.get("crowding", "Crowding_below_4")
        overbite = profile.get("overbite", "Normal_overbite")
        overjet = profile.get("overjet", "Normal_overjet")

        # 1. Molar relationship shift (Class II/III)
        mal_params = MALOCCLUSION_GEOMETRY.get(malocclusion, MALOCCLUSION_GEOMETRY["ClassI"])
        molar_shift = mal_params["molar_shift_mm"]
        if abs(molar_shift) > 0.01:
            # Upper first molars (indices 5, 12 = teeth 16, 26)
            # Shift anteriorly (positive Y direction in our arch layout)
            for idx in [5, 12]:  # upper molar_1 positions
                config[idx, 5] += molar_shift  # Y shift
            # For Class III, also shift lower molars
            if molar_shift < 0:
                for idx in [19, 26]:  # lower molar_1 positions
                    config[idx, 5] += abs(molar_shift)

        # 2. Overjet (horizontal protrusion of upper incisors)
        oj_params = OVERJET_PARAMS.get(overjet, OVERJET_PARAMS["Normal_overjet"])
        protrusion = oj_params["horizontal_protrusion_mm"]
        if protrusion > 1.5:
            # Upper incisors (indices 0,1,7,8 = teeth 11,12,21,22)
            for idx in [0, 1, 7, 8]:
                config[idx, 5] -= protrusion  # push forward (negative Y = labial)

        # 3. Overbite (vertical overlap)
        ob_params = OVERBITE_PARAMS.get(overbite, OVERBITE_PARAMS["Normal_overbite"])
        vert_overlap = ob_params["vertical_overlap_mm"]
        if vert_overlap > 2.5:
            # Upper incisors drop down, lower incisors rise up
            for idx in [0, 1, 7, 8]:
                config[idx, 6] -= vert_overlap * 0.5  # Z down
            for idx in [14, 15, 21, 22]:  # lower incisors
                config[idx, 6] += vert_overlap * 0.5  # Z up

        # 4. Crowding (arch compression)
        cr_params = CROWDING_PARAMS.get(crowding, CROWDING_PARAMS["Crowding_below_4"])
        compression = cr_params["arch_compression"]
        if compression > 0.03:
            # Compress arch: scale X coordinates toward center, add rotational crowding
            for i in range(N_TEETH):
                config[i, 4] *= (1.0 - compression)
                # Add slight rotation for crowded teeth
                if rng.random() < compression * 5:  # ~50% for 10% compression
                    rot_deg = rng.uniform(5.0, 15.0)
                    axis = np.array([0.0, 0.0, 1.0])  # tipping
                    delta_q = quaternion_from_axis_angle(axis, math.radians(rot_deg))
                    old_q = config[i, :4]
                    config[i, :4] = quaternion_normalize(quaternion_multiply(delta_q, old_q))

        # 5. Add random noise on top (scaled by difficulty)
        difficulty_level = profile.get("difficulty_level", "easy")
        config = self.apply_malocclusion(config, difficulty_level, rng)

        return config

    def generate_case_for_profile(
        self,
        profile: Dict[str, Any],
        seed: int,
    ) -> Dict[str, Any]:
        """
        Generate a dental case informed by a real clinical profile.
        """
        rng = np.random.default_rng(seed)
        ideal = self.generate_ideal_config()
        initial = self.apply_clinical_perturbation(ideal, profile, rng)
        baseline_traj = self.generate_baseline_trajectory(initial, ideal)

        return {
            "initial_config": initial,
            "target_config": ideal,
            "tooth_ids": TOOTH_IDS,
            "tooth_types": TOOTH_TYPES,
            "baseline_trajectory": baseline_traj,
            "difficulty": profile.get("difficulty_level", "easy"),
            "seed": seed,
            "clinical_profile": profile,
        }

    def apply_adversarial_jitter(
        self,
        trajectory: np.ndarray,
        current_stage: int,
        jitter_strength: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Apply SE(3) jitter to trajectory[current_stage].
        jitter_strength: 0.1 = mild (0.1mm, 1deg), 0.3 = strong (0.3mm, 3deg)
        """
        perturbed = trajectory.copy()

        # Select 1-4 teeth to jitter
        n_jitter = int(rng.integers(1, 5))
        tooth_indices = rng.choice(N_TEETH, size=n_jitter, replace=False)

        for idx in tooth_indices:
            # Translation jitter
            trans_noise = rng.standard_normal(3) * jitter_strength
            perturbed[current_stage, idx, 4:7] += trans_noise

            # Rotation jitter
            max_angle_deg = jitter_strength * 10.0  # 0.1 -> 1 deg, 0.3 -> 3 deg
            perturbed[current_stage, idx, :4] = random_quaternion_perturbation(
                perturbed[current_stage, idx, :4],
                max_angle_deg,
                rng,
            )

        return perturbed, tooth_indices.tolist()
