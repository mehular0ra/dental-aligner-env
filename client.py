"""
battisiBot HTTP client for the Dental Aligner Trajectory Planning environment.

Usage:
    from client import DentalAlignerEnvClient
    env = DentalAlignerEnvClient(base_url='http://localhost:7860')
    obs = env.reset(task_id='task_easy', seed=42)
    # Build your 24-stage trajectory...
    obs = env.step(trajectory=stages, reasoning='SLERP plan', confidence=0.8)
"""
import json
from typing import Any, Dict, List, Optional

import requests


class DentalAlignerEnvClient:
    """
    HTTP client for the Dental Aligner Trajectory Planning environment.

    Wraps /reset and /step HTTP endpoints with a clean Python API.
    Compatible with battisiBot inference agent and custom agents.
    """

    def __init__(self, base_url: str = 'http://localhost:7860', timeout: float = 60.0):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self._last_obs: Optional[Dict[str, Any]] = None

    def reset(
        self,
        task_id: str = 'task_easy',
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        model_name: str = 'battisiBot',
    ) -> Dict[str, Any]:
        """
        Reset the environment and return the initial observation.

        Parameters
        ----------
        task_id : str
            One of 'task_easy', 'task_medium', 'task_hard'.
        seed : int, optional
            Random seed for reproducibility.
        episode_id : str, optional
            Custom episode identifier.
        model_name : str
            Agent/model name for logging.

        Returns
        -------
        dict
            Observation dict with keys: tooth_table, tooth_table_text, task_description,
            baseline_trajectory_json, arch_graph_json, current_stage, stages_remaining, ...
        """
        payload = {
            'task_id': task_id,
            'model_name': model_name,
        }
        if seed is not None:
            payload['seed'] = seed
        if episode_id is not None:
            payload['episode_id'] = episode_id

        resp = requests.post(
            f'{self.base_url}/reset',
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        self._last_obs = data.get('observation', {})
        return data

    def step(
        self,
        trajectory: List[Dict[str, Any]],
        reasoning: str = '',
        confidence: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Submit a planned trajectory and receive graded feedback.

        Parameters
        ----------
        trajectory : list of dicts
            Each dict: {'stage_index': int, 'tooth_ids': list, 'poses': list[list[float]]}
            - stage_index: 1-24 for full plan
            - tooth_ids: list of 28 FDI tooth IDs
            - poses: list of 28 poses, each [qw, qx, qy, qz, tx, ty, tz]
        reasoning : str
            Agent's planning rationale (shown in feedback).
        confidence : float
            Agent's confidence in the plan [0.0, 1.0].

        Returns
        -------
        dict with keys: observation, reward (float), done (bool)
        """
        payload = {
            'action': {
                'trajectory': trajectory,
                'reasoning': reasoning,
                'confidence': confidence,
                'metadata': {},
            }
        }
        resp = requests.post(
            f'{self.base_url}/step',
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        self._last_obs = data.get('observation', {})
        return data

    def health(self) -> Dict[str, str]:
        """Check server health."""
        resp = requests.get(f'{self.base_url}/health', timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_tasks(self) -> List[Dict[str, Any]]:
        """List available tasks."""
        resp = requests.get(f'{self.base_url}/tasks', timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get('tasks', [])

    def get_constraints(self) -> Dict[str, Any]:
        """Return clinical movement constraints."""
        resp = requests.get(f'{self.base_url}/constraints', timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_grader_info(self, task_id: str) -> Dict[str, Any]:
        """Return grader weight information for a task."""
        resp = requests.get(f'{self.base_url}/grade/{task_id}', timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    @property
    def last_observation(self) -> Optional[Dict[str, Any]]:
        """Return the last observation received."""
        return self._last_obs
