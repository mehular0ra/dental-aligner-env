#!/usr/bin/env python3
"""
Evaluation harness for the dental aligner GRPO pipeline.

Runs three policies over a held-out evaluation set and reports a
per-component breakdown:

  1. SLERP_NODECAY  — straight-line baseline graded on the planned trajectory
  2. SLERP_DECAY    — same plan, graded on the decay-realised trajectory
  3. AGENT_DECAY    — agent-generated plan graded on the decay-realised trajectory
                      (model loaded if --model is passed and transformers is installed;
                       otherwise scripted-agent fallback that varies tooth_plans by
                       difficulty so we can sanity-check the pipeline end-to-end)

Outputs:
  - artifacts/eval_results.json : full per-case scores and aggregate stats
  - artifacts/eval_summary.md   : table with means / stds / deltas
  - artifacts/reward_curve.png  : per-difficulty bar chart (if matplotlib available)

Usage:
  python eval_grpo.py                                # scripted-agent fallback
  python eval_grpo.py --n 50                         # 50 cases per difficulty
  python eval_grpo.py --model checkpoints/final --n 50  # trained model
"""
import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.dental_constants import TOOTH_IDS, TOOTH_TYPES
from server.synthetic_data import DentalCaseGenerator
from server.grader import AlignerGrader
from server.force_decay import apply_force_decay
from server.clinical_profiles import sample_profile

from train_grpo import (
    format_observation_prompt,
    parse_llm_output_to_trajectory,
)


def grade_components(
    grader: AlignerGrader,
    traj: np.ndarray,
    initial: np.ndarray,
    target: np.ndarray,
    task_id: str,
) -> Dict[str, float]:
    accuracy = grader.compute_final_accuracy(traj, target)["final_accuracy"]
    smoothness = grader.compute_smoothness(traj)
    compliance = grader.compute_constraint_compliance(traj)["compliance_score"]
    staging = grader.compute_staging_quality(traj, initial, target)
    total, _ = grader.grade(task_id, traj, initial, target)
    return {
        "total": float(total),
        "accuracy": float(accuracy),
        "smoothness": float(smoothness),
        "compliance": float(compliance),
        "staging": float(staging),
    }


def _scripted_agent_completion(case: dict, profile: Optional[dict]) -> str:
    """
    A scripted "agent" that imitates clinical staging priority. Used when no
    real LLM is available — gives us a non-trivial baseline to compare against
    SLERP and a hook for the eval harness to be exercised on CPU.

    Strategy: incisors start at stage 1, premolars at 4, molars at 8.
    """
    # tooth_type → (start_stage, end_stage)
    schedule = {
        "central_incisor":  (1, 16),
        "lateral_incisor":  (1, 18),
        "canine":           (3, 20),
        "first_premolar":   (5, 22),
        "second_premolar":  (6, 23),
        "first_molar":      (8, 24),
        "second_molar":     (10, 24),
    }
    plans = []
    for tid in TOOTH_IDS:
        ttype = TOOTH_TYPES[tid]
        start, end = schedule.get(ttype, (1, 24))
        plans.append({
            "tooth_id": tid, "start_stage": start, "end_stage": end,
            "ease_in": 0.3, "ease_out": 0.2,
        })
    return json.dumps({"tooth_plans": plans})


def _load_model_if_available(model_path: Optional[str]):
    if not model_path:
        return None, None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        tok = AutoTokenizer.from_pretrained(model_path)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
        )
        mdl.eval()
        return mdl, tok
    except Exception as e:
        print(f"[eval] Could not load model from {model_path}: {e}")
        print("[eval] Falling back to scripted agent.")
        return None, None


def _llm_completion(model, tok, prompt: str, max_tokens: int = 1024) -> str:
    import torch
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False, temperature=1.0,
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text


def evaluate(
    n_per_difficulty: int = 25,
    model_path: Optional[str] = None,
    output_dir: str = "artifacts",
    seed_base: int = 9000,
) -> dict:
    case_gen = DentalCaseGenerator()
    grader = AlignerGrader()
    model, tok = _load_model_if_available(model_path)

    results = {"slerp_nodecay": [], "slerp_decay": [], "agent_decay": []}
    difficulties = ["easy", "medium", "hard"]

    for diff in difficulties:
        for i in range(n_per_difficulty):
            seed = seed_base + difficulties.index(diff) * 1000 + i
            rng = np.random.default_rng(seed)
            try:
                profile = sample_profile(diff, rng)
                case = case_gen.generate_case_for_profile(profile, seed)
            except Exception:
                case = case_gen.generate_case(diff, seed)
                profile = None

            initial = case["initial_config"]
            target = case["target_config"]
            slerp = case["baseline_trajectory"]

            # 1) SLERP without decay
            r1 = grade_components(grader, slerp, initial, target, f"task_{diff}")
            r1.update({"difficulty": diff, "seed": seed})
            results["slerp_nodecay"].append(r1)

            # 2) SLERP with decay
            slerp_actual = apply_force_decay(slerp, initial)
            r2 = grade_components(grader, slerp_actual, initial, target, f"task_{diff}")
            r2.update({"difficulty": diff, "seed": seed})
            results["slerp_decay"].append(r2)

            # 3) Agent (LLM or scripted) with decay
            prompt = format_observation_prompt(case, profile)
            if model is not None:
                completion = _llm_completion(model, tok, prompt)
            else:
                completion = _scripted_agent_completion(case, profile)
            agent_traj = parse_llm_output_to_trajectory(completion, initial, target)
            agent_actual = apply_force_decay(agent_traj, initial)
            r3 = grade_components(grader, agent_actual, initial, target, f"task_{diff}")
            r3.update({"difficulty": diff, "seed": seed})
            results["agent_decay"].append(r3)

    # Aggregate by difficulty
    summary = {}
    for policy, rows in results.items():
        summary[policy] = {}
        for diff in difficulties:
            subset = [r for r in rows if r["difficulty"] == diff]
            if not subset:
                continue
            agg = {}
            for k in ["total", "accuracy", "smoothness", "compliance", "staging"]:
                vals = np.array([r[k] for r in subset])
                agg[k] = {"mean": float(vals.mean()), "std": float(vals.std()), "n": len(vals)}
            summary[policy][diff] = agg

    # Write artifacts
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump({"results": results, "summary": summary,
                   "n_per_difficulty": n_per_difficulty,
                   "model_path": model_path or "scripted_agent"}, f, indent=2)

    # Markdown summary
    md_lines = ["# GRPO Evaluation Summary\n",
                f"- Cases per difficulty: {n_per_difficulty}",
                f"- Policy under test: {model_path or 'scripted_agent (no model)'}",
                ""]
    md_lines.append("| Difficulty | Policy | Total | Accuracy | Smoothness | Compliance | Staging |")
    md_lines.append("|------------|--------|-------|----------|------------|------------|---------|")
    for diff in difficulties:
        for policy in ["slerp_nodecay", "slerp_decay", "agent_decay"]:
            if diff not in summary.get(policy, {}):
                continue
            agg = summary[policy][diff]
            md_lines.append(
                f"| {diff} | {policy} | "
                f"{agg['total']['mean']:.4f}±{agg['total']['std']:.3f} | "
                f"{agg['accuracy']['mean']:.4f} | "
                f"{agg['smoothness']['mean']:.4f} | "
                f"{agg['compliance']['mean']:.4f} | "
                f"{agg['staging']['mean']:.4f} |"
            )
    md_lines.append("")
    md_lines.append("## Deltas (agent_decay vs slerp_decay)")
    md_lines.append("| Difficulty | Δtotal | Δaccuracy | Δstaging |")
    md_lines.append("|------------|--------|-----------|----------|")
    for diff in difficulties:
        if diff not in summary.get("agent_decay", {}) or diff not in summary.get("slerp_decay", {}):
            continue
        a = summary["agent_decay"][diff]
        s = summary["slerp_decay"][diff]
        md_lines.append(
            f"| {diff} | {a['total']['mean'] - s['total']['mean']:+.4f} | "
            f"{a['accuracy']['mean'] - s['accuracy']['mean']:+.4f} | "
            f"{a['staging']['mean'] - s['staging']['mean']:+.4f} |"
        )
    with open(os.path.join(output_dir, "eval_summary.md"), "w") as f:
        f.write("\n".join(md_lines))

    # Bar chart (best-effort)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(difficulties))
        width = 0.27
        for k, policy in enumerate(["slerp_nodecay", "slerp_decay", "agent_decay"]):
            means = [summary[policy][d]["total"]["mean"] if d in summary.get(policy, {}) else 0 for d in difficulties]
            stds = [summary[policy][d]["total"]["std"] if d in summary.get(policy, {}) else 0 for d in difficulties]
            ax.bar(x + (k - 1) * width, means, width, yerr=stds, label=policy, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(difficulties)
        ax.set_ylabel("Total reward")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("SLERP vs decay vs agent — per-difficulty mean reward")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "reward_curve.png"), dpi=140)
        plt.close(fig)
    except ImportError:
        pass

    return summary


def main():
    parser = argparse.ArgumentParser(description="Eval harness for dental GRPO")
    parser.add_argument("--n", type=int, default=25, help="Cases per difficulty")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model (optional)")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--seed-base", type=int, default=9000)
    args = parser.parse_args()

    print(f"[eval] Running {args.n} cases per difficulty (model={args.model or 'scripted'})")
    summary = evaluate(args.n, args.model, args.output_dir, args.seed_base)

    # Console table
    print("\n[eval] Summary:")
    print(f"  {'difficulty':<10} {'policy':<16} {'total':>8} {'acc':>8} {'staging':>8}")
    for diff in ["easy", "medium", "hard"]:
        for policy in ["slerp_nodecay", "slerp_decay", "agent_decay"]:
            if diff not in summary.get(policy, {}):
                continue
            agg = summary[policy][diff]
            print(f"  {diff:<10} {policy:<16} {agg['total']['mean']:>8.4f} "
                  f"{agg['accuracy']['mean']:>8.4f} {agg['staging']['mean']:>8.4f}")

    print(f"\n[eval] Wrote {args.output_dir}/eval_results.json, eval_summary.md, reward_curve.png")


if __name__ == "__main__":
    main()
