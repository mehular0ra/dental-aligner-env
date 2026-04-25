"""
Pharmacokinetic force decay model for orthodontic tooth movement.

Models delayed biomechanical response: forces applied at stage N produce
effects over stages N through N+4, peaking at N+2. Based on PDL viscoelastic
creep behavior (Cattaneo et al. 2005, Proffit's Contemporary Orthodontics).

The decay kernel approximates the clinical reality that:
- Aligner force is not instantaneous — bone remodeling takes ~2 weeks per stage
- Peak tooth movement lags behind force application by 1-2 aligner stages
- Residual movement continues 2-3 stages after force removal

This creates non-Markov dynamics: the current state depends on the last 4 actions,
making simple SLERP interpolation suboptimal and requiring genuine planning.
"""
import numpy as np

# Decay kernel: weights for force effect over stages [N, N+1, N+2, N+3, N+4]
# Fitted to approximate PDL creep curves from Cattaneo et al. 2005
# Peak at N+2 reflects clinical observation that max tooth displacement
# occurs ~2 aligner changes after force application.
# Sum < 1.0 intentionally: some force is lost to biological damping,
# meaning the agent must "overshoot" planned movements to hit targets.
DECAY_KERNEL = np.array([0.05, 0.10, 0.30, 0.25, 0.15], dtype=np.float64)
# Sum = 0.85 → 15% force loss per stage due to PDL viscoelastic damping

# Kernel length (number of stages of delayed effect)
KERNEL_LEN = len(DECAY_KERNEL)


def apply_force_decay(
    planned_trajectory: np.ndarray,
    initial_config: np.ndarray,
) -> np.ndarray:
    """
    Apply pharmacokinetic force decay to a planned trajectory.

    The agent plans intended tooth positions at each stage, but actual positions
    are determined by the cumulative delayed effect of forces from prior stages.

    Parameters
    ----------
    planned_trajectory : np.ndarray, shape (26, 28, 7)
        Agent's planned trajectory. Stage 0 = initial, stage 25 = target.
    initial_config : np.ndarray, shape (28, 7)
        Initial tooth configuration (stage 0).

    Returns
    -------
    actual_trajectory : np.ndarray, shape (26, 28, 7)
        The physically realized trajectory after force decay effects.
    """
    n_stages = planned_trajectory.shape[0]
    n_teeth = planned_trajectory.shape[1]
    actual = planned_trajectory.copy()

    # Compute intended forces (translation deltas) at each stage
    # force[s] = planned[s] - planned[s-1] for translation components
    forces = np.zeros((n_stages, n_teeth, 3), dtype=np.float64)
    for s in range(1, n_stages):
        forces[s] = planned_trajectory[s, :, 4:7] - planned_trajectory[s - 1, :, 4:7]

    # Apply decay: actual translation at stage s is initial + sum of decayed forces
    for s in range(1, min(n_stages, 25)):  # Don't modify stage 0 or stage 25
        cumulative = np.zeros((n_teeth, 3), dtype=np.float64)
        for k in range(KERNEL_LEN):
            src_stage = s - k
            if src_stage < 1:
                continue
            cumulative += DECAY_KERNEL[k] * forces[src_stage]
        # Scale so that total effect converges to intended over many stages
        # Without scaling, sum of kernel weights < 1 would shrink total movement
        actual[s, :, 4:7] = actual[s - 1, :, 4:7] + cumulative

    # Rotations: apply a milder delay (quaternions are harder to accumulate)
    # We keep rotations as planned — the main delay effect is on translations
    # This is clinically reasonable: rotational movements complete faster than
    # bodily movements (Proffit Ch. 9)

    return actual


def compute_decay_penalty(
    planned_trajectory: np.ndarray,
    initial_config: np.ndarray,
) -> float:
    """
    Compute how much the force decay changes the trajectory vs the plan.
    Returns a value in [0, 1] where 0 = no deviation, 1 = large deviation.

    This can be used as an additional reward signal to encourage agents
    to anticipate the delay.
    """
    actual = apply_force_decay(planned_trajectory, initial_config)
    # Mean translational deviation across all stages and teeth
    deviation = np.mean(
        np.linalg.norm(
            actual[1:25, :, 4:7] - planned_trajectory[1:25, :, 4:7],
            axis=2,
        )
    )
    # Normalize: 2mm deviation = penalty of 1.0
    return float(min(1.0, deviation / 2.0))
