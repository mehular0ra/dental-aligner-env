"""
FastAPI application for the Dental Aligner Trajectory Planning Environment.

Uses a persistent shared environment instance for HTTP endpoints
(reset + step are always sequential in evaluation).

Bonus endpoints: /tasks, /grade/{task_id}, /constraints
"""

import json
from typing import Any, Dict, Optional

from fastapi import Body
from fastapi.responses import JSONResponse

from openenv.core.env_server import create_fastapi_app

from models import AlignerAction, AlignerObservation
from server.dental_environment import DentalAlignerEnvironment, _SESSIONS
from server.dental_constants import (
    TOOTH_IDS, N_TEETH, N_STAGES,
    MAX_TRANSLATION_PER_STAGE_MM, MAX_ROTATION_PER_STAGE_DEG,
)

# ---------------------------------------------------------------------------
# Shared singleton environment
# ---------------------------------------------------------------------------
_shared_env = DentalAlignerEnvironment()

# Module-level variable: set in reset_override, read in step_override
_current_model_name: str = 'unknown'


def _env_factory() -> DentalAlignerEnvironment:
    """Return the shared environment instance (not a new one per request)."""
    return _shared_env


# Build the base app using the openenv factory
app = create_fastapi_app(_env_factory, AlignerAction, AlignerObservation)


# ---------------------------------------------------------------------------
# Override /reset to accept task_id and model_name in body
# ---------------------------------------------------------------------------

@app.post('/reset', include_in_schema=False)
async def reset_override(request: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """Reset endpoint: accepts optional task_id, seed, episode_id, model_name."""
    global _current_model_name

    task_id: Optional[str]    = request.get('task_id', None)
    seed: Optional[int]       = request.get('seed', None)
    episode_id: Optional[str] = request.get('episode_id', None)
    _current_model_name       = request.get('model_name', 'unknown')

    obs = _shared_env.reset(seed=seed, episode_id=episode_id, task_id=task_id)

    obs_dict = obs.model_dump(exclude={'reward', 'done', 'metadata'})
    return {
        'observation': obs_dict,
        'reward': obs.reward,
        'done': obs.done,
    }


# ---------------------------------------------------------------------------
# Override /step
# ---------------------------------------------------------------------------

@app.post('/step', include_in_schema=False)
async def step_override(request: Dict[str, Any]) -> Dict[str, Any]:
    """Step endpoint. Returns graded observation with reward."""
    global _current_model_name

    action_data = request.get('action', {})
    action = AlignerAction.model_validate(action_data)
    obs = _shared_env.step(action)

    obs_dict = obs.model_dump(exclude={'reward', 'done', 'metadata'})
    return {
        'observation': obs_dict,
        'reward': obs.reward,
        'done': obs.done,
    }


# ---------------------------------------------------------------------------
# Bonus endpoints
# ---------------------------------------------------------------------------

@app.get('/health')
async def health() -> Dict[str, str]:
    """Health check."""
    return {'status': 'healthy'}


@app.get('/state')
async def get_state() -> Dict[str, Any]:
    """Return the current environment state as a dict."""
    state = _shared_env.state
    return state.model_dump()


@app.get('/tasks')
async def list_tasks():
    """List all available tasks."""
    return JSONResponse({'tasks': [
        {
            'id': 'task_easy',
            'difficulty': 'easy',
            'n_teeth_perturbed': '4-6',
            'description': (
                'Easy malocclusion: 4-6 teeth displaced 1-3 mm, '
                '5-15° z-axis rotation. Single-step episode.'
            ),
        },
        {
            'id': 'task_medium',
            'difficulty': 'medium',
            'n_teeth_perturbed': '10-14',
            'description': (
                'Medium malocclusion: 10-14 teeth displaced 2-5 mm, '
                '10-20° multi-axis rotation. Single-step episode.'
            ),
        },
        {
            'id': 'task_hard',
            'difficulty': 'hard',
            'n_teeth_perturbed': '18-24',
            'description': (
                'Hard malocclusion: 18-24 teeth displaced 3-8 mm, '
                '15-25° combined motion. Two-step episode with adversarial jitter.'
            ),
        },
    ]})


@app.get('/constraints')
async def get_constraints():
    """Return clinical movement constraints enforced by the grader."""
    return JSONResponse({
        'max_translation_per_stage_mm': MAX_TRANSLATION_PER_STAGE_MM,
        'max_rotation_per_stage_deg':   MAX_ROTATION_PER_STAGE_DEG,
        'n_stages':                     N_STAGES,
        'n_teeth':                      N_TEETH,
        'tooth_ids':                    TOOTH_IDS,
    })


@app.get('/grade/{task_id}')
async def get_grader_info(task_id: str):
    """Return grader weight information for a given task."""
    info = {
        'task_easy': (
            'Scoring components: final_accuracy 50% + smoothness 25% + compliance 25%. '
            'Easy difficulty: 4-6 teeth perturbed. '
            'Baseline SLERP scores ~0.40.'
        ),
        'task_medium': (
            'Scoring components: final_accuracy 50% + smoothness 25% + compliance 25%. '
            'Medium difficulty: 10-14 teeth perturbed, multi-axis rotations. '
            'Baseline SLERP scores ~0.40.'
        ),
        'task_hard': (
            'Scoring components: final_accuracy 50% + smoothness 25% + compliance 10% + recovery 15%. '
            'Hard difficulty: 18-24 teeth perturbed. Two-step episode. '
            'Adversarial jitter injected after step 1. '
            'Recovery quality contributes 15% to the final score.'
        ),
    }
    return JSONResponse({
        'task_id': task_id,
        'grader': info.get(task_id, f'Unknown task_id: {task_id}'),
        'components': {
            'final_accuracy': '50%',
            'smoothness':     '25%',
            'compliance':     '25% (10% for task_hard)',
            'recovery':       '15% (task_hard only)',
        },
    })


def main():
    """Entry point for uvicorn server."""
    import uvicorn
    uvicorn.run('server.app:app', host='0.0.0.0', port=7860, reload=False)


if __name__ == '__main__':
    main()
