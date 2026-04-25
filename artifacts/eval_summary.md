# GRPO Evaluation Summary

- Cases per difficulty: 8
- Policy under test: scripted_agent (no model)

| Difficulty | Policy | Total | Accuracy | Smoothness | Compliance | Staging |
|------------|--------|-------|----------|------------|------------|---------|
| easy | slerp_nodecay | 0.9446±0.014 | 0.9761 | 0.9601 | 0.9955 | 0.8153 |
| easy | slerp_decay | 0.8761±0.020 | 0.8815 | 0.8169 | 1.0000 | 0.8008 |
| easy | agent_decay | 0.9093±0.014 | 0.9285 | 0.8838 | 0.9946 | 0.8112 |
| medium | slerp_nodecay | 0.8990±0.010 | 0.9554 | 0.9046 | 0.9821 | 0.6113 |
| medium | slerp_decay | 0.7408±0.023 | 0.7805 | 0.4788 | 1.0000 | 0.6255 |
| medium | agent_decay | 0.8111±0.015 | 0.8599 | 0.6617 | 0.9470 | 0.6826 |
| hard | slerp_nodecay | 0.3333±0.013 | 0.9129 | 0.8119 | 0.6786 | 0.5191 |
| hard | slerp_decay | 0.2216±0.013 | 0.5900 | 0.0000 | 0.8536 | 0.5272 |
| hard | agent_decay | 0.2454±0.016 | 0.7270 | 0.0597 | 0.7262 | 0.5478 |

## Deltas (agent_decay vs slerp_decay)
| Difficulty | Δtotal | Δaccuracy | Δstaging |
|------------|--------|-----------|----------|
| easy | +0.0332 | +0.0470 | +0.0105 |
| medium | +0.0703 | +0.0794 | +0.0571 |
| hard | +0.0239 | +0.1370 | +0.0206 |