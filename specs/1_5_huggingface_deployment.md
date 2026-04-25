# 1.5 — HuggingFace Spaces Deployment

> **Priority:** VERY IMPORTANT
> **Estimated Effort:** 1-2 hours
> **Judge Score Delta:** +1.0/10 (gates hackathon submission requirements)
> **Primary Judging Criterion:** Pipeline (10%, deployability)

---

## 1. Impact

### Hackathon Impact
- Submission gate: HF Spaces URL is required. No deploy = invalid submission.
- Judges WILL `curl` `/health`, `/reset_stepwise`, `/step_stepwise`. Anything other than 200/JSON sinks credibility instantly.

### Research Impact
- Public, reproducible env URL → cited as the reference implementation.

### Demo Value
- One-line: "Every result in this pitch is reproducible by `curl https://<space>.hf.space/step`."

---

## 2. Scientific Foundation
- N/A — infra only.

---

## 3. Implementation Details

### Architecture
```
Dockerfile  →  pip install requirements.txt
            →  uvicorn server.app:app --host 0.0.0.0 --port 7860
HF Spaces  →  docker SDK, port 7860
GET /health  → {status: ok, env: dental-aligner-env}
POST /reset  → AlignerObservation
POST /step   → AlignerObservation (with reward, done)
```

### Files Changed
| File | Change Type | Description |
|------|------------|-------------|
| `Dockerfile` | VERIFY | already exists; check it copies all `server/`, `models.py`, `datasets/tsinghua/case_database.json` |
| `requirements.txt` | VERIFY | numpy, scipy, fastapi, uvicorn, pydantic, openenv |
| `openenv.yaml` | VERIFY | metadata required by HF Spaces |
| `README.md` (front-matter) | DONE | sdk: docker, app_port: 7860 |
| `.dockerignore` | NEW (recommended) | exclude `.venv`, `*.rar`, `landmarks/Landmark_annotation/` if too large |

### Smoke-Test Script
```bash
SPACE_URL=https://battisibot-dental-aligner-env.hf.space  # update to your space
curl -fsS $SPACE_URL/health | jq .
curl -fsS -X POST $SPACE_URL/reset -H 'content-type: application/json' \
     -d '{"task_id":"task_easy"}' | jq '.task_id, .stages_remaining, .done'
# Submit a SLERP-shaped trajectory (use scripts/post_slerp.py)
python scripts/post_slerp.py $SPACE_URL task_medium
```

---

## 4. Prerequisites
- 1_1, 1_2, 1_3, 1_4 should be merged so the live env reflects the spec.
- HF account with Pro subscription (for ZeroGPU, optional for CPU-only env serving).
- `huggingface_hub` CLI for `huggingface-cli login`.

---

## 5. Validation

### Unit Tests
- Local docker build: `docker build -t dental-env . && docker run -p 7860:7860 dental-env` — `/health` returns 200.
- `curl localhost:7860/reset -X POST -d '{"task_id":"task_easy"}'` returns valid `AlignerObservation`.

### Integration Tests
- After push: HF Spaces logs show "Application startup complete" within 5 minutes.
- Smoke-test script above passes against the live space.

### Success Criteria
- Quantitative: latency `/step` < 5 s on free CPU tier.
- Qualitative: `inference.py` runs end-to-end against the live space.
- Demo: pitch slide includes the live space URL as a clickable link; in rehearsal, we live-curl `/health` on screen.

---

## 6. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| `case_database.json` (3 MB) bloats image | LOW | slow build | already accepted, file is small relative to base image |
| `landmarks/Landmark_annotation/` (10s of MB) ships unintentionally | MED | huge image | add to `.dockerignore` |
| `numpy`/`scipy` wheel install fails on HF builder | LOW | deploy fail | pin known-good versions in `requirements.txt`; fallback path uses pure-Python in `inference.py` (already done) |
| HF Space sleeps → first call cold (60 s) | HIGH | judge sees timeout | add a Pre-PR warm-up curl in submission notes; or use `huggingface_hub` keepalive cron |

---

## 7. Q&A Defense

| Judge Question | Prepared Answer |
|----------------|----------------|
| "Can I run this locally?" | "Yes — `docker build -t dental-env . && docker run -p 7860:7860 dental-env`. Same image as the live HF space." |
| "Why HF Spaces, not Render/Modal?" | "HF Spaces is the OpenEnv-recommended host; integrates with `huggingface_hub` and the OpenEnv client out of the box." |

---

## 8. Pitch Integration
- 0:30: "Live at https://battisibot-dental-aligner-env.hf.space — try it now."
- Last slide: smoke-test command on screen.
