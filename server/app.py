"""
FastAPI application for the Dental Aligner Trajectory Planning Environment.

Endpoints:
  GET  /          — HTML dashboard with GIF visualizer
  POST /reset     — Start a new episode
  POST /step      — Submit a trajectory action
  POST /demo_run  — Run SLERP vs clinical-staged demo, return GIFs + scores
  GET  /health    — Health check
  GET  /tasks     — List tasks
  GET  /constraints
  GET  /grade/{task_id}
  GET  /state
"""

import json
from typing import Any, Dict, Optional

import numpy as np
from fastapi import Body
from fastapi.responses import JSONResponse, HTMLResponse

from openenv.core.env_server import create_fastapi_app

from models import AlignerAction, AlignerObservation
from server.dental_environment import DentalAlignerEnvironment, _SESSIONS
from server.dental_constants import (
    TOOTH_IDS, TOOTH_TYPES, N_TEETH, N_STAGES,
    MAX_TRANSLATION_PER_STAGE_MM, MAX_ROTATION_PER_STAGE_DEG,
)
from server.synthetic_data import DentalCaseGenerator
from server.grader import AlignerGrader
from server.quaternion_utils import quaternion_slerp, quaternion_normalize

# ---------------------------------------------------------------------------
# Shared singleton environment
# ---------------------------------------------------------------------------
_shared_env = DentalAlignerEnvironment()
_current_model_name: str = 'unknown'


def _env_factory() -> DentalAlignerEnvironment:
    return _shared_env


app = create_fastapi_app(_env_factory, AlignerAction, AlignerObservation)


# ---------------------------------------------------------------------------
# Demo helpers — build trajectories without touching env state
# ---------------------------------------------------------------------------

def _build_staged_slerp(
    initial_config: np.ndarray,
    target_config: np.ndarray,
) -> np.ndarray:
    """Build a clinical-priority SLERP trajectory (no LLM, deterministic).

    Incisors move first (stages 1-10), molars last (stages 13-24).
    Returns (26, 28, 7) array.
    """
    STAGING = {
        'central_incisor': (1,  10),
        'lateral_incisor': (2,  12),
        'canine':          (4,  14),
        'premolar_1':      (7,  18),
        'premolar_2':      (8,  19),
        'molar_1':         (13, 23),
        'molar_2':         (14, 24),
    }

    traj = np.zeros((26, N_TEETH, 7), dtype=np.float64)
    traj[0]  = initial_config.copy()
    traj[25] = target_config.copy()

    for i, tid in enumerate(TOOTH_IDS):
        tt = TOOTH_TYPES[tid]
        start, end = STAGING.get(tt, (1, 24))
        q_init = quaternion_normalize(initial_config[i, :4])
        t_init = initial_config[i, 4:]
        q_tgt  = quaternion_normalize(target_config[i, :4])
        t_tgt  = target_config[i, 4:]

        for s in range(1, 25):
            if s < start:
                alpha = 0.0
            elif s > end:
                alpha = 1.0
            else:
                alpha = (s - start) / max(1.0, end - start)
            # Smoothstep easing
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)

            traj[s, i, :4] = quaternion_normalize(quaternion_slerp(q_init, q_tgt, alpha))
            traj[s, i, 4:] = (1.0 - alpha) * t_init + alpha * t_tgt

    return traj


# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>battisiBot — Dental Aligner Trajectory Planner</title>
  <style>
    :root {
      --blue:   #1B6CA8;
      --teal:   #0D9488;
      --orange: #EA580C;
      --bg:     #F0F4F8;
      --card:   #FFFFFF;
      --border: #D1D9E0;
      --text:   #1A202C;
      --muted:  #64748B;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); }

    /* ---- header ---- */
    header {
      background: linear-gradient(135deg, var(--blue) 0%, #0D4F8C 100%);
      color: white; padding: 20px 32px;
      display: flex; align-items: center; gap: 16px;
    }
    header .logo { font-size: 2rem; }
    header h1 { font-size: 1.5rem; font-weight: 700; }
    header p  { font-size: 0.85rem; opacity: 0.8; margin-top: 2px; }

    /* ---- main layout ---- */
    main { max-width: 1100px; margin: 28px auto; padding: 0 20px; }

    /* ---- card ---- */
    .card {
      background: var(--card); border: 1px solid var(--border);
      border-radius: 12px; padding: 20px 24px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.07);
      margin-bottom: 20px;
    }
    .card h2 { font-size: 1rem; font-weight: 600; color: var(--muted); text-transform: uppercase;
               letter-spacing: .05em; margin-bottom: 14px; }

    /* ---- task tabs ---- */
    .tab-row { display: flex; gap: 10px; flex-wrap: wrap; }
    .tab-btn {
      padding: 8px 22px; border: 2px solid var(--border); border-radius: 8px;
      background: white; cursor: pointer; font-size: 0.92rem; font-weight: 500;
      transition: all .15s; color: var(--muted);
    }
    .tab-btn:hover  { border-color: var(--blue); color: var(--blue); }
    .tab-btn.active { background: var(--blue); border-color: var(--blue); color: white; }
    .tab-btn.medium.active { background: var(--teal); border-color: var(--teal); }
    .tab-btn.hard.active   { background: var(--orange); border-color: var(--orange); }

    /* ---- controls row ---- */
    .ctrl-row { display: flex; gap: 14px; align-items: flex-end; flex-wrap: wrap; margin-top: 14px; }
    .field label { display: block; font-size: 0.8rem; font-weight: 500; color: var(--muted);
                   margin-bottom: 4px; }
    .field input {
      border: 1.5px solid var(--border); border-radius: 7px; padding: 7px 12px;
      font-size: 0.9rem; width: 100px; outline: none;
    }
    .field input:focus { border-color: var(--blue); }
    #run-btn {
      background: var(--blue); color: white; border: none; border-radius: 8px;
      padding: 9px 26px; font-size: 0.95rem; font-weight: 600; cursor: pointer;
      transition: background .15s;
    }
    #run-btn:hover:not(:disabled) { background: #155A8C; }
    #run-btn:disabled { opacity: 0.55; cursor: not-allowed; }

    /* ---- score cards ---- */
    .scores-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }
    .score-card {
      border-radius: 10px; padding: 14px 16px; text-align: center;
      border: 1px solid var(--border);
    }
    .score-card .label { font-size: 0.72rem; text-transform: uppercase; font-weight: 600;
                         color: var(--muted); letter-spacing: .06em; margin-bottom: 4px; }
    .score-card .value { font-size: 1.6rem; font-weight: 700; }
    .score-card.total { background: var(--blue); border-color: var(--blue); color: white; }
    .score-card.total .label { color: rgba(255,255,255,.75); }
    .score-card.staged { background: var(--teal); border-color: var(--teal); color: white; }
    .score-card.staged .label { color: rgba(255,255,255,.75); }

    /* ---- diff badge ---- */
    .diff-badge {
      display: inline-block; font-size: 0.75rem; font-weight: 600;
      padding: 2px 8px; border-radius: 99px; margin-left: 6px;
    }
    .diff-badge.pos { background: #D1FAE5; color: #065F46; }
    .diff-badge.neg { background: #FEE2E2; color: #991B1B; }

    /* ---- GIF section ---- */
    .gif-tabs { display: flex; gap: 8px; margin-bottom: 14px; }
    .gif-tab {
      padding: 6px 16px; border-radius: 20px; border: 1.5px solid var(--border);
      cursor: pointer; font-size: 0.85rem; font-weight: 500; color: var(--muted);
      transition: all .15s;
    }
    .gif-tab.active { border-color: var(--blue); background: var(--blue); color: white; }
    .gif-pane { display: none; }
    .gif-pane.active { display: block; }
    .gif-pane img {
      max-width: 100%; border-radius: 8px; border: 1px solid var(--border);
      display: block; margin: 0 auto;
    }

    /* ---- spinner ---- */
    #spinner {
      display: none; text-align: center; padding: 40px;
      color: var(--muted); font-size: 0.95rem;
    }
    .spinner-ring {
      width: 40px; height: 40px; border: 4px solid var(--border);
      border-top-color: var(--blue); border-radius: 50%;
      animation: spin .8s linear infinite; margin: 0 auto 12px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ---- results hidden initially ---- */
    #results { display: none; }

    /* ---- feedback box ---- */
    #feedback-box {
      background: #F8FAFC; border: 1px solid var(--border); border-radius: 8px;
      padding: 12px 16px; font-family: monospace; font-size: 0.78rem;
      white-space: pre-wrap; line-height: 1.55; max-height: 200px; overflow-y: auto;
      color: var(--text);
    }

    /* ---- tooth table ---- */
    #tooth-table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    th { background: var(--bg); font-weight: 600; color: var(--muted); padding: 7px 10px;
         text-align: left; border-bottom: 1px solid var(--border); }
    td { padding: 6px 10px; border-bottom: 1px solid #EEF2F7; }
    tr:hover td { background: #F8FAFC; }

    /* ---- footer ---- */
    footer { text-align: center; padding: 28px; color: var(--muted); font-size: 0.8rem; }
    footer a { color: var(--blue); text-decoration: none; }
  </style>
</head>
<body>

<header>
  <div class="logo">🦷</div>
  <div>
    <h1>battisiBot &nbsp;&mdash;&nbsp; Dental Aligner Trajectory Planner</h1>
    <p>OpenEnv · SE(3) trajectory planning · 28 teeth · 24 aligner stages</p>
  </div>
</header>

<main>

  <!-- Controls -->
  <div class="card">
    <h2>Configure Demo</h2>
    <div class="tab-row">
      <button class="tab-btn easy  active" data-task="task_easy">Easy</button>
      <button class="tab-btn medium"       data-task="task_medium">Medium</button>
      <button class="tab-btn hard"         data-task="task_hard">Hard</button>
    </div>
    <div class="ctrl-row">
      <div class="field">
        <label>Random Seed</label>
        <input type="number" id="seed-input" value="42" min="0" max="99999">
      </div>
      <button id="run-btn">▶&nbsp; Visualize</button>
    </div>
  </div>

  <!-- Spinner -->
  <div id="spinner">
    <div class="spinner-ring"></div>
    Generating 26-stage trajectory and rendering GIF animation…
  </div>

  <!-- Results -->
  <div id="results">

    <!-- Score cards -->
    <div class="card" id="scores-card">
      <h2>Scores</h2>
      <div class="scores-grid" id="scores-grid"></div>
    </div>

    <!-- GIF Viewer -->
    <div class="card">
      <h2>Trajectory Visualization</h2>
      <div class="gif-tabs">
        <div class="gif-tab active" data-pane="single">SLERP Baseline</div>
        <div class="gif-tab"       data-pane="staged">Clinical Staged</div>
        <div class="gif-tab"       data-pane="compare">Side-by-Side</div>
      </div>
      <div id="pane-single" class="gif-pane active">
        <img id="gif-single" src="" alt="SLERP Baseline GIF">
      </div>
      <div id="pane-staged" class="gif-pane">
        <img id="gif-staged" src="" alt="Clinical Staged GIF">
      </div>
      <div id="pane-compare" class="gif-pane">
        <img id="gif-compare" src="" alt="Comparison GIF">
      </div>
    </div>

    <!-- Grader feedback -->
    <div class="card">
      <h2>Grader Feedback</h2>
      <pre id="feedback-box">—</pre>
    </div>

    <!-- Tooth table -->
    <div class="card">
      <h2>Tooth Movement Summary (most displaced)</h2>
      <div id="tooth-table-wrap">
        <table>
          <thead>
            <tr>
              <th>Tooth</th><th>Type</th>
              <th>Δ Trans (mm)</th><th>Δ Rot (°)</th>
              <th>Initial Tx</th><th>Target Tx</th>
            </tr>
          </thead>
          <tbody id="tooth-tbody"></tbody>
        </table>
      </div>
    </div>

  </div><!-- /results -->
</main>

<footer>
  battisiBot &nbsp;|&nbsp; OpenEnv Round 1 &nbsp;|&nbsp;
  <a href="/docs" target="_blank">API Docs</a> &nbsp;|&nbsp;
  <a href="/health" target="_blank">Health</a>
</footer>

<script>
  let selectedTask = 'task_easy';

  // Tab selection
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      selectedTask = btn.dataset.task;
    });
  });

  // GIF pane tabs
  document.querySelectorAll('.gif-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.gif-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.gif-pane').forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById('pane-' + tab.dataset.pane).classList.add('active');
    });
  });

  document.getElementById('run-btn').addEventListener('click', runDemo);

  async function runDemo() {
    const seed = parseInt(document.getElementById('seed-input').value) || 42;
    const runBtn = document.getElementById('run-btn');
    runBtn.disabled = true;

    document.getElementById('results').style.display = 'none';
    document.getElementById('spinner').style.display  = 'block';

    try {
      const resp = await fetch('/demo_run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_id: selectedTask, seed }),
      });
      if (!resp.ok) throw new Error('Server error: ' + resp.status);
      const data = await resp.json();
      renderResults(data);
    } catch (err) {
      alert('Error: ' + err.message);
    } finally {
      document.getElementById('spinner').style.display = 'none';
      runBtn.disabled = false;
    }
  }

  function pct(v) { return (v * 100).toFixed(1) + '%'; }
  function fmt(v) { return typeof v === 'number' ? v.toFixed(4) : '—'; }

  function renderResults(d) {
    // Score cards
    const sGrid = document.getElementById('scores-grid');
    const s = d.slerp_scores;
    const a = d.staged_scores;
    const diff = (a.total - s.total);
    const diffHtml = `<span class="diff-badge ${diff >= 0 ? 'pos' : 'neg'}">${diff >= 0 ? '+' : ''}${(diff*100).toFixed(1)}%</span>`;

    sGrid.innerHTML = `
      <div class="score-card total">
        <div class="label">SLERP Total</div>
        <div class="value">${pct(s.total)}</div>
      </div>
      <div class="score-card staged">
        <div class="label">Staged Total ${diffHtml}</div>
        <div class="value">${pct(a.total)}</div>
      </div>
      <div class="score-card">
        <div class="label">Accuracy (SLERP)</div>
        <div class="value">${pct(s.accuracy)}</div>
      </div>
      <div class="score-card">
        <div class="label">Accuracy (Staged)</div>
        <div class="value">${pct(a.accuracy)}</div>
      </div>
      <div class="score-card">
        <div class="label">Smoothness</div>
        <div class="value">${pct(s.smoothness)}</div>
      </div>
      <div class="score-card">
        <div class="label">Compliance</div>
        <div class="value">${pct(s.compliance)}</div>
      </div>
      <div class="score-card">
        <div class="label">Staging Quality</div>
        <div class="value">${pct(a.staging)}</div>
      </div>
    `;

    // GIFs
    if (d.gif_slerp)   document.getElementById('gif-single').src  = 'data:image/gif;base64,' + d.gif_slerp;
    if (d.gif_staged)  document.getElementById('gif-staged').src  = 'data:image/gif;base64,' + d.gif_staged;
    if (d.gif_compare) document.getElementById('gif-compare').src = 'data:image/gif;base64,' + d.gif_compare;

    // Feedback
    document.getElementById('feedback-box').textContent =
      (d.slerp_feedback || '') + '\\n\\n' + (d.staged_feedback || '');

    // Tooth table (top 12 most displaced)
    const tbody = document.getElementById('tooth-tbody');
    tbody.innerHTML = '';
    (d.tooth_movements || []).slice(0, 14).forEach(t => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td><b>${t.id}</b></td><td>${t.type}</td>
        <td>${t.delta_trans.toFixed(2)}</td><td>${t.delta_rot.toFixed(1)}</td>
        <td>${t.init_tx.toFixed(2)}</td><td>${t.tgt_tx.toFixed(2)}</td>`;
      tbody.appendChild(tr);
    });

    document.getElementById('results').style.display = 'block';
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
  }
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# /  — HTML Dashboard
# ---------------------------------------------------------------------------

@app.get('/', response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    return HTMLResponse(content=_DASHBOARD_HTML)


# ---------------------------------------------------------------------------
# /demo_run  — Generate GIFs + scores without touching env state
# ---------------------------------------------------------------------------

@app.post('/demo_run')
async def demo_run(request: Dict[str, Any] = Body(default={})):
    """Run a demo episode using SLERP baseline and clinical-staged SLERP.

    Returns GIF animations (base64) and score breakdowns.
    Does NOT advance the env state — uses a fresh internal case generator.
    """
    from server.visualization import (
        trajectory_to_gif_base64,
        trajectory_to_gif_base64_single,
        generate_comparison_gif_base64,
    )

    task_id  = request.get('task_id', 'task_easy')
    seed     = int(request.get('seed', 42))
    difficulty_map = {'task_easy': 'easy', 'task_medium': 'medium', 'task_hard': 'hard'}
    difficulty = difficulty_map.get(task_id, 'easy')

    # Generate fresh case
    gen    = DentalCaseGenerator()
    grader = AlignerGrader()
    case   = gen.generate_case(difficulty, seed)

    initial = case['initial_config']   # (28, 7)
    target  = case['target_config']    # (28, 7)
    slerp   = case['baseline_trajectory']  # (26, 28, 7)

    # Clinical-priority staged SLERP
    staged = _build_staged_slerp(initial, target)

    # Grade both trajectories
    slerp_reward,  slerp_fb  = grader.grade(task_id, slerp,  initial, target)
    staged_reward, staged_fb = grader.grade(task_id, staged, initial, target)

    # Score component breakdown (grader internals)
    def _components(traj):
        acc  = grader.compute_final_accuracy(traj, target)['final_accuracy']
        smo  = grader.compute_smoothness(traj)
        comp = grader.compute_constraint_compliance(traj)['compliance_score']
        stag = grader.compute_staging_quality(traj, initial, target)
        return dict(accuracy=round(acc, 4), smoothness=round(smo, 4),
                    compliance=round(comp, 4), staging=round(stag, 4),
                    total=round(slerp_reward if traj is slerp else staged_reward, 4))

    slerp_scores  = _components(slerp)
    staged_scores = _components(staged)
    staged_scores['total'] = round(staged_reward, 4)

    # Build GIFs
    gif_slerp   = trajectory_to_gif_base64_single(slerp,  label='SLERP | ')
    gif_staged  = trajectory_to_gif_base64_single(staged, label='Staged | ')
    gif_compare = generate_comparison_gif_base64(slerp, staged)

    # Tooth movement summary (sorted by displacement)
    movements = []
    for i, tid in enumerate(TOOTH_IDS):
        delta_t = float(np.linalg.norm(target[i, 4:] - initial[i, 4:]))
        from server.quaternion_utils import quaternion_multiply, quaternion_inverse, quaternion_to_angle_deg
        q_rel   = quaternion_multiply(target[i, :4], quaternion_inverse(initial[i, :4]))
        delta_r = float(quaternion_to_angle_deg(q_rel))
        movements.append({
            'id':         tid,
            'type':       TOOTH_TYPES[tid],
            'delta_trans': round(delta_t, 3),
            'delta_rot':   round(delta_r, 3),
            'init_tx':     round(float(initial[i, 4]), 2),
            'tgt_tx':      round(float(target[i, 4]),  2),
        })
    movements.sort(key=lambda m: -m['delta_trans'])

    return JSONResponse({
        'task_id':       task_id,
        'difficulty':    difficulty,
        'seed':          seed,
        'slerp_scores':  slerp_scores,
        'staged_scores': staged_scores,
        'slerp_feedback':  slerp_fb,
        'staged_feedback': staged_fb,
        'gif_slerp':     gif_slerp,
        'gif_staged':    gif_staged,
        'gif_compare':   gif_compare,
        'tooth_movements': movements,
    })


# ---------------------------------------------------------------------------
# Override /reset to accept task_id and model_name in body
# ---------------------------------------------------------------------------

@app.post('/reset', include_in_schema=False)
async def reset_override(request: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    global _current_model_name

    task_id:    Optional[str] = request.get('task_id', None)
    seed:       Optional[int] = request.get('seed', None)
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
    return {'status': 'healthy'}


@app.get('/state')
async def get_state() -> Dict[str, Any]:
    return _shared_env.state.model_dump()


@app.get('/tasks')
async def list_tasks():
    return JSONResponse({'tasks': [
        {
            'id': 'task_easy',
            'difficulty': 'easy',
            'n_teeth_perturbed': '4-6',
            'description': 'Easy: 4-6 teeth displaced 1-3 mm, 5-15° z-axis rotation.',
        },
        {
            'id': 'task_medium',
            'difficulty': 'medium',
            'n_teeth_perturbed': '10-14',
            'description': 'Medium: 10-14 teeth displaced 2-5 mm, 10-20° multi-axis rotation.',
        },
        {
            'id': 'task_hard',
            'difficulty': 'hard',
            'n_teeth_perturbed': '18-24',
            'description': 'Hard: 18-24 teeth displaced 3-8 mm. Two-step with adversarial jitter.',
        },
    ]})


@app.get('/constraints')
async def get_constraints():
    return JSONResponse({
        'max_translation_per_stage_mm': MAX_TRANSLATION_PER_STAGE_MM,
        'max_rotation_per_stage_deg':   MAX_ROTATION_PER_STAGE_DEG,
        'n_stages': N_STAGES,
        'n_teeth':  N_TEETH,
        'tooth_ids': TOOTH_IDS,
    })


@app.get('/grade/{task_id}')
async def get_grader_info(task_id: str):
    info = {
        'task_easy':   'final_accuracy 40% + smoothness 20% + compliance 20% + staging 20%',
        'task_medium': 'final_accuracy 45% + smoothness 20% + compliance 20% + staging 15%',
        'task_hard':   'final_accuracy 40% + smoothness 15% + compliance 15% + staging 15% + recovery 15%',
    }
    return JSONResponse({
        'task_id': task_id,
        'grader': info.get(task_id, f'Unknown task_id: {task_id}'),
    })


def main():
    import uvicorn
    uvicorn.run('server.app:app', host='0.0.0.0', port=7860, reload=False)


if __name__ == '__main__':
    main()
