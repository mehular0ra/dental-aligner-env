#!/usr/bin/env python3
"""
Preparation script for the Dental Aligner Trajectory Planning environment.

Validates datasets, checks dependencies, and prepares the environment for
training and evaluation.

Usage:
  python prepare.py           # Full validation
  python prepare.py --quick   # Quick check only
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def check_dataset():
    """Validate Tsinghua case database is accessible."""
    db_path = os.path.join(ROOT, "datasets", "tsinghua", "case_database.json")
    if not os.path.exists(db_path):
        print(f"  WARN: {db_path} not found")
        return False
    with open(db_path) as f:
        data = json.load(f)
    print(f"  OK: case_database.json loaded ({len(data)} patient profiles)")
    return True


def check_server_modules():
    """Validate all server modules import correctly."""
    sys.path.insert(0, ROOT)
    modules = [
        "server.dental_constants",
        "server.quaternion_utils",
        "server.synthetic_data",
        "server.grader",
        "server.clinical_profiles",
        "server.force_decay",
        "server.dental_environment",
    ]
    ok = True
    for mod in modules:
        try:
            __import__(mod)
            print(f"  OK: {mod}")
        except Exception as e:
            print(f"  FAIL: {mod} — {e}")
            ok = False
    return ok


def check_environment():
    """Run a quick environment smoke test."""
    from server.dental_environment import DentalAlignerEnvironment
    env = DentalAlignerEnvironment()
    obs = env.reset(seed=42, task_id="task_easy")
    assert not obs.done, "Episode should not be done after reset"
    assert obs.stages_remaining == 24, f"Expected 24 stages, got {obs.stages_remaining}"
    has_profile = "CLINICAL PROFILE" in obs.task_description
    print(f"  OK: Environment reset works (clinical_profile={has_profile})")
    return True


def check_force_decay():
    """Verify force decay produces meaningful score drop."""
    import numpy as np
    from server.synthetic_data import DentalCaseGenerator
    from server.grader import AlignerGrader
    from server.force_decay import apply_force_decay

    gen = DentalCaseGenerator()
    grader = AlignerGrader()
    case = gen.generate_case("medium", seed=42)
    initial, target = case["initial_config"], case["target_config"]
    baseline = case["baseline_trajectory"]

    r_no_decay, _ = grader.grade("task_medium", baseline, initial, target)
    actual = apply_force_decay(baseline, initial)
    r_decay, _ = grader.grade("task_medium", actual, initial, target)

    drop = r_no_decay - r_decay
    print(f"  OK: Force decay drops SLERP score by {drop:.4f} ({r_no_decay:.4f} → {r_decay:.4f})")
    return drop > 0.01


def check_training_pipeline():
    """Verify training pipeline imports and reward function works."""
    try:
        from train_grpo import (
            format_observation_prompt,
            parse_llm_output_to_trajectory,
            compute_reward,
            generate_training_prompts,
            make_reward_fns,
        )
        from server.synthetic_data import DentalCaseGenerator
        from server.grader import AlignerGrader

        gen = DentalCaseGenerator()
        grader = AlignerGrader()

        prompts = generate_training_prompts(3, gen)
        assert len(prompts) == 3
        print(f"  OK: Training prompt generation works ({len(prompts)} prompts)")

        # Test reward computation
        case = gen.generate_case("easy", seed=42)
        traj = case["baseline_trajectory"]
        r = compute_reward(traj, case["initial_config"], case["target_config"], "task_easy", grader)
        print(f"  OK: Reward computation works (baseline={r:.4f})")

        # Multi-reward functions
        fns = make_reward_fns(gen, grader)
        assert len(fns) == 3, "expected 3 per-component reward fns"
        import json as _json
        comp = _json.dumps({"tooth_plans": []})
        for fn in fns:
            r = fn([comp], prompts=[prompts[0]["prompt"]])
            assert isinstance(r, list) and len(r) == 1
        print(f"  OK: Multi-reward functions return per-component scores ({[f.__name__ for f in fns]})")
        return True
    except Exception as e:
        print(f"  FAIL: Training pipeline — {e}")
        return False


def check_eval_harness():
    """Verify the eval harness runs end-to-end on a tiny sample."""
    try:
        from eval_grpo import evaluate
        summary = evaluate(n_per_difficulty=2, model_path=None,
                           output_dir="artifacts/_smoke", seed_base=12345)
        # Scripted agent must beat SLERP+decay on at least one difficulty
        beats = []
        for diff in ["easy", "medium", "hard"]:
            if diff in summary["agent_decay"] and diff in summary["slerp_decay"]:
                d = summary["agent_decay"][diff]["total"]["mean"] - summary["slerp_decay"][diff]["total"]["mean"]
                beats.append((diff, d))
        beats_str = ", ".join(f"{d}:{v:+.3f}" for d, v in beats)
        print(f"  OK: Eval harness runs ({beats_str})")
        return any(v > 0 for _, v in beats)
    except Exception as e:
        print(f"  FAIL: Eval harness — {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick check only")
    args = parser.parse_args()

    print("=" * 60)
    print("Dental Aligner Environment — Preparation Check")
    print("=" * 60)

    checks = [
        ("Dataset", check_dataset),
        ("Server modules", check_server_modules),
        ("Environment", check_environment),
        ("Force decay", check_force_decay),
    ]

    if not args.quick:
        checks.append(("Training pipeline", check_training_pipeline))
        checks.append(("Eval harness", check_eval_harness))

    results = []
    for name, fn in checks:
        print(f"\n[{name}]")
        try:
            ok = fn()
            results.append((name, ok))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Summary:")
    all_ok = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            all_ok = False

    print("=" * 60)
    if all_ok:
        print("All checks passed. Ready for training/evaluation.")
    else:
        print("Some checks failed. See details above.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
