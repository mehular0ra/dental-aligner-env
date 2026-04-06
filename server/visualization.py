"""
Dental trajectory visualization — generates GIF animations showing tooth
positions and orientations at each aligner stage.

Each frame = one stage (0=initial through 25=final).
Teeth are shown as oriented ellipses in a 2D top-down view of the dental arch.
Color-coding: upper arch=blue, lower arch=orange. Arrow shows tooth orientation.
"""
import io
import json
import math
import base64
from typing import List, Optional, Dict, Any

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Ellipse, FancyArrowPatch
    from PIL import Image
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False


TOOTH_IDS = [11,12,13,14,15,16,17, 21,22,23,24,25,26,27,
             31,32,33,34,35,36,37, 41,42,43,44,45,46,47]

UPPER_IDS = set([11,12,13,14,15,16,17, 21,22,23,24,25,26,27])
LOWER_IDS = set([31,32,33,34,35,36,37, 41,42,43,44,45,46,47])

MOLAR_IDS    = set([16,17,26,27,36,37,46,47])
PREMOLAR_IDS = set([14,15,24,25,34,35,44,45])
CANINE_IDS   = set([13,23,33,43])
INCISOR_IDS  = set([11,12,21,22,31,32,41,42])

TOOTH_COLORS = {
    'upper': '#4A90D9',
    'lower': '#E8894A',
}

TOOTH_TYPE_SIZES = {
    'central_incisor': (4.0, 5.5),
    'lateral_incisor': (3.5, 5.0),
    'canine':          (3.5, 5.5),
    'premolar_1':      (4.5, 5.0),
    'premolar_2':      (4.5, 5.0),
    'molar_1':         (6.0, 5.5),
    'molar_2':         (5.5, 5.0),
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def quaternion_to_yaw_deg(q: np.ndarray) -> float:
    """Extract yaw (rotation about z-axis) from quaternion [qw, qx, qy, qz].

    yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

    Returns degrees.
    """
    qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    yaw_rad = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz)
    )
    return math.degrees(yaw_rad)


def get_tooth_type(tooth_id: int) -> str:
    """Return tooth type string based on FDI number.

    Quadrant digit is the tens place; position digit is the units place.
    Position 1 = central incisor, 2 = lateral incisor, 3 = canine,
    4 = first premolar, 5 = second premolar, 6 = first molar, 7 = second molar.
    """
    position = tooth_id % 10
    type_map = {
        1: 'central_incisor',
        2: 'lateral_incisor',
        3: 'canine',
        4: 'premolar_1',
        5: 'premolar_2',
        6: 'molar_1',
        7: 'molar_2',
    }
    return type_map.get(position, 'central_incisor')


# ---------------------------------------------------------------------------
# Single-frame renderer
# ---------------------------------------------------------------------------

def render_stage_frame(
    config: np.ndarray,
    stage_num: int,
    ax,
    title_prefix: str = '',
    arch: str = 'upper',
) -> None:
    """Render one stage of teeth onto a matplotlib Axes.

    Parameters
    ----------
    config:
        Shape (28, 7). One row per tooth in TOOTH_IDS order.
        Each row: [qw, qx, qy, qz, tx, ty, tz].
    stage_num:
        Stage index 0-25 used only for the title.
    ax:
        Matplotlib Axes to draw on.
    title_prefix:
        Optional string prepended to the frame title.
    arch:
        'upper' or 'lower' — which set of teeth to render.
    """
    ax.cla()

    ids_to_render = UPPER_IDS if arch == 'upper' else LOWER_IDS
    color = TOOTH_COLORS[arch]
    arch_label = 'Upper Arch' if arch == 'upper' else 'Lower Arch'

    for idx, tooth_id in enumerate(TOOTH_IDS):
        if tooth_id not in ids_to_render:
            continue

        row = config[idx]
        qw, qx, qy, qz = row[0], row[1], row[2], row[3]
        tx, ty = float(row[4]), float(row[5])

        yaw_deg = quaternion_to_yaw_deg(np.array([qw, qx, qy, qz]))

        tooth_type = get_tooth_type(tooth_id)
        width_mm, height_mm = TOOTH_TYPE_SIZES[tooth_type]
        # Scale down for display
        w = width_mm * 0.6
        h = height_mm * 0.6

        ellipse = Ellipse(
            xy=(tx, ty),
            width=w,
            height=h,
            angle=yaw_deg,
            linewidth=1.2,
            edgecolor='black',
            facecolor=color,
            alpha=0.75,
            zorder=2,
        )
        ax.add_patch(ellipse)

        # Draw orientation arrow
        arrow_len = h * 0.5
        angle_rad = math.radians(yaw_deg)
        dx = arrow_len * math.sin(angle_rad)
        dy = arrow_len * math.cos(angle_rad)
        ax.annotate(
            '',
            xy=(tx + dx, ty + dy),
            xytext=(tx, ty),
            arrowprops=dict(
                arrowstyle='->', color='white', lw=1.5
            ),
            zorder=3,
        )

        # Tooth ID label
        ax.text(
            tx, ty,
            str(tooth_id),
            ha='center', va='center',
            fontsize=5.5,
            fontweight='bold',
            color='white',
            zorder=4,
        )

    # Axis formatting
    ax.set_xlim(-45, 45)
    ax.set_ylim(-45, 45)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_xlabel('X (mm)', fontsize=8)
    ax.set_ylabel('Y (mm)', fontsize=8)

    stage_label = f'Stage {stage_num:02d}/25'
    full_title = f'{title_prefix}{arch_label} — {stage_label}'
    ax.set_title(full_title, fontsize=9, pad=4)

    # Color legend patch
    patch = mpatches.Patch(facecolor=color, edgecolor='black', label=arch_label)
    ax.legend(handles=[patch], loc='lower right', fontsize=7)


# ---------------------------------------------------------------------------
# GIF assembly helpers
# ---------------------------------------------------------------------------

def _fig_to_pil(fig) -> "Image.Image":
    """Convert a matplotlib figure to a PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=90, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img.convert('RGB')


def _build_frames(
    trajectory: np.ndarray,
    prefix: str = '',
) -> List["Image.Image"]:
    """Build one PIL Image per stage from a (26, 28, 7) trajectory array."""
    n_stages = trajectory.shape[0]
    frames: List[Image.Image] = []

    for stage_idx in range(n_stages):
        config = trajectory[stage_idx]  # (28, 7)

        fig, (ax_upper, ax_lower) = plt.subplots(1, 2, figsize=(10, 8))
        fig.subplots_adjust(wspace=0.3)

        render_stage_frame(config, stage_idx, ax_upper, title_prefix=prefix, arch='upper')
        render_stage_frame(config, stage_idx, ax_lower, title_prefix=prefix, arch='lower')

        img = _fig_to_pil(fig)
        plt.close(fig)
        frames.append(img)

    return frames


def _add_pause_frames(
    frames: List["Image.Image"],
    n_start: int = 3,
    n_end: int = 3,
) -> List["Image.Image"]:
    """Duplicate first/last frames to create a visual pause."""
    if not frames:
        return frames
    start_pause = [frames[0].copy() for _ in range(n_start)]
    end_pause = [frames[-1].copy() for _ in range(n_end)]
    return start_pause + frames + end_pause


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def trajectory_to_gif(
    trajectory: np.ndarray,
    output_path: str,
    fps: int = 4,
    initial_config: Optional[np.ndarray] = None,
    target_config: Optional[np.ndarray] = None,
) -> str:
    """Generate a GIF animation from a full trajectory array.

    Parameters
    ----------
    trajectory:
        Shape (26, 28, 7). Stages 0-25 inclusive.
    output_path:
        File path where the .gif will be saved.
    fps:
        Frames per second for the animation.
    initial_config:
        Unused — kept for API compatibility. Stage 0 of trajectory is used.
    target_config:
        Unused — kept for API compatibility. Stage 25 of trajectory is used.

    Returns
    -------
    str
        The output_path that was written.
    """
    if not HAS_VISUALIZATION:
        return output_path

    frames = _build_frames(trajectory, prefix='')
    frames = _add_pause_frames(frames, n_start=3, n_end=3)

    duration_ms = int(1000 / fps)
    frames[0].save(
        output_path,
        format='GIF',
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return output_path


def trajectory_to_gif_base64_single(
    trajectory: np.ndarray,
    label: str = '',
    fps: int = 4,
) -> str:
    """Single-trajectory GIF as base64 string with an optional label prefix."""
    if not HAS_VISUALIZATION:
        return ''
    frames = _build_frames(trajectory, prefix=label)
    frames = _add_pause_frames(frames, n_start=3, n_end=5)
    duration_ms = int(1000 / fps)
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format='GIF',
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def trajectory_to_gif_base64(
    trajectory: np.ndarray,
    fps: int = 4,
) -> str:
    """Like trajectory_to_gif but returns a base64-encoded GIF string.

    Parameters
    ----------
    trajectory:
        Shape (26, 28, 7).
    fps:
        Frames per second.

    Returns
    -------
    str
        Base64-encoded GIF bytes, or an error dict string if unavailable.
    """
    if not HAS_VISUALIZATION:
        return json.dumps({'error': 'visualization not available'})

    frames = _build_frames(trajectory, prefix='')
    frames = _add_pause_frames(frames, n_start=3, n_end=3)

    duration_ms = int(1000 / fps)
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format='GIF',
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_comparison_gif(
    baseline_trajectory: np.ndarray,
    agent_trajectory: np.ndarray,
    output_path: str,
    fps: int = 4,
) -> str:
    """Generate a side-by-side comparison GIF: SLERP baseline vs agent.

    Parameters
    ----------
    baseline_trajectory:
        Shape (26, 28, 7). SLERP interpolated trajectory.
    agent_trajectory:
        Shape (26, 28, 7). Agent-planned trajectory.
    output_path:
        File path where the .gif will be saved.
    fps:
        Frames per second.

    Returns
    -------
    str
        The output_path that was written.
    """
    if not HAS_VISUALIZATION:
        return output_path

    n_stages = min(baseline_trajectory.shape[0], agent_trajectory.shape[0])
    duration_ms = int(1000 / fps)
    frames: List[Image.Image] = []

    for stage_idx in range(n_stages):
        base_config = baseline_trajectory[stage_idx]
        agent_config = agent_trajectory[stage_idx]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.subplots_adjust(wspace=0.3, hspace=0.4)
        fig.suptitle(f'Stage {stage_idx:02d}/25 — SLERP Baseline vs Agent', fontsize=11)

        ax_base_upper, ax_agent_upper = axes[0][0], axes[0][1]
        ax_base_lower, ax_agent_lower = axes[1][0], axes[1][1]

        render_stage_frame(base_config,  stage_idx, ax_base_upper,  title_prefix='SLERP | ', arch='upper')
        render_stage_frame(agent_config, stage_idx, ax_agent_upper, title_prefix='Agent  | ', arch='upper')
        render_stage_frame(base_config,  stage_idx, ax_base_lower,  title_prefix='SLERP | ', arch='lower')
        render_stage_frame(agent_config, stage_idx, ax_agent_lower, title_prefix='Agent  | ', arch='lower')

        img = _fig_to_pil(fig)
        plt.close(fig)
        frames.append(img)

    frames = _add_pause_frames(frames, n_start=3, n_end=3)

    frames[0].save(
        output_path,
        format='GIF',
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return output_path


def generate_comparison_gif_base64(
    baseline_trajectory: np.ndarray,
    agent_trajectory: np.ndarray,
    fps: int = 3,
) -> str:
    """Side-by-side comparison GIF (SLERP baseline vs clinical staged) as base64."""
    if not HAS_VISUALIZATION:
        return ''

    n_stages = min(baseline_trajectory.shape[0], agent_trajectory.shape[0])
    duration_ms = int(1000 / fps)
    frames: List[Image.Image] = []

    for stage_idx in range(n_stages):
        base_config  = baseline_trajectory[stage_idx]
        agent_config = agent_trajectory[stage_idx]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.subplots_adjust(wspace=0.3, hspace=0.4)
        fig.suptitle(
            f'Stage {stage_idx:02d} / 25  —  SLERP Baseline (left)  vs  Clinical Staged (right)',
            fontsize=11, fontweight='bold',
        )

        render_stage_frame(base_config,  stage_idx, axes[0][0], title_prefix='SLERP | ', arch='upper')
        render_stage_frame(agent_config, stage_idx, axes[0][1], title_prefix='Staged | ', arch='upper')
        render_stage_frame(base_config,  stage_idx, axes[1][0], title_prefix='SLERP | ', arch='lower')
        render_stage_frame(agent_config, stage_idx, axes[1][1], title_prefix='Staged | ', arch='lower')

        img = _fig_to_pil(fig)
        plt.close(fig)
        frames.append(img)

    frames = _add_pause_frames(frames, n_start=4, n_end=6)

    buf = io.BytesIO()
    frames[0].save(
        buf,
        format='GIF',
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
