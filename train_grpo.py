#!/usr/bin/env python3
"""
GRPO Training Pipeline for Dental Aligner Trajectory Planning.

Uses embedded environment (no HTTP server) for fast reward computation.
Trains Qwen2.5-1.5B with Unsloth 4-bit QLoRA via TRL GRPOTrainer.

Usage:
  # Quick test (10 steps)
  python train_grpo.py --steps 10

  # Full training run
  python train_grpo.py --steps 300 --wandb

  # Resume from checkpoint
  python train_grpo.py --steps 300 --resume checkpoints/latest
"""

import argparse
import json
import math
import os
import sys
import time
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None

# ---------------------------------------------------------------------------
# Dental environment imports (embedded, no HTTP)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from server.dental_constants import (
    TOOTH_IDS, TOOTH_TYPES, N_TEETH, N_STAGES,
    MAX_TRANSLATION_PER_STAGE_MM, MAX_ROTATION_PER_STAGE_DEG,
)
from server.synthetic_data import DentalCaseGenerator
from server.grader import AlignerGrader
from server.force_decay import apply_force_decay
from server.clinical_profiles import sample_profile
from server.quaternion_utils import quaternion_slerp, quaternion_normalize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "checkpoints"
WANDB_PROJECT = "orthorl-grpo"

# Prompt template for the LLM
PROMPT_TEMPLATE = """You are an orthodontic treatment planner. Plan tooth movements for 24 aligner stages.

PATIENT: {clinical_info}
DIFFICULTY: {difficulty}

TOOTH DATA (tooth_id: remaining_trans_mm, remaining_rot_deg):
{tooth_summary}

OUTPUT: JSON with per-tooth interpolation parameters.
{{"tooth_plans": [{{"tooth_id": N, "start_stage": S, "end_stage": E, "ease_in": 0.3, "ease_out": 0.3}}...]}}
RULES: Incisors start early (1-4), canines next (3-7), premolars middle (6-14), molars last (12-20).
Teeth with remaining_trans < 0.5mm: start=1, end=24."""


def format_observation_prompt(case: dict, profile: Optional[dict] = None) -> str:
    """Format a dental case into a compact prompt for the LLM."""
    initial = case["initial_config"]
    target = case["target_config"]

    # Build tooth summary
    lines = []
    for i, tid in enumerate(TOOTH_IDS):
        trans_dist = float(np.linalg.norm(initial[i, 4:7] - target[i, 4:7]))
        # Simplified rotation distance
        q_dot = abs(float(np.dot(initial[i, :4], target[i, :4])))
        rot_deg = 2 * math.degrees(math.acos(min(1.0, q_dot))) if q_dot < 1.0 else 0.0
        ttype = TOOTH_TYPES[tid]
        lines.append(f"  {tid}({ttype[:3]}): {trans_dist:.1f}mm, {rot_deg:.1f}°")

    tooth_summary = "\n".join(lines)

    clinical_info = "Unknown"
    difficulty = case.get("difficulty", "easy")
    if profile:
        clinical_info = (
            f"{profile.get('malocclusion', '?')}, "
            f"{profile.get('crowding', '?')}, "
            f"{profile.get('overbite', '?')}, "
            f"{profile.get('overjet', '?')}"
        )

    return PROMPT_TEMPLATE.format(
        clinical_info=clinical_info,
        difficulty=difficulty,
        tooth_summary=tooth_summary,
    )


def parse_llm_output_to_trajectory(
    output_text: str,
    initial_config: np.ndarray,
    target_config: np.ndarray,
) -> np.ndarray:
    """
    Parse LLM JSON output into a (26, 28, 7) trajectory.
    Falls back to SLERP baseline if parsing fails.
    """
    trajectory = np.zeros((26, N_TEETH, 7), dtype=np.float64)
    trajectory[0] = initial_config.copy()
    trajectory[25] = target_config.copy()

    # Try to parse JSON from output
    tooth_plans = None
    try:
        # Find JSON in output
        text = output_text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            tooth_plans = data.get("tooth_plans", [])
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # Build per-tooth interpolation plan
    plan_map = {}
    if tooth_plans:
        for tp in tooth_plans:
            tid = tp.get("tooth_id")
            if tid is not None:
                plan_map[tid] = tp

    # Generate trajectory using SLERP + per-tooth timing
    for i, tid in enumerate(TOOTH_IDS):
        q0 = initial_config[i, :4]
        q1 = target_config[i, :4]
        t0 = initial_config[i, 4:7]
        t1 = target_config[i, 4:7]

        plan = plan_map.get(tid, {})
        start_stage = max(1, min(24, plan.get("start_stage", 1)))
        end_stage = max(start_stage, min(24, plan.get("end_stage", 24)))
        ease_in = max(0.0, min(1.0, plan.get("ease_in", 0.0)))
        ease_out = max(0.0, min(1.0, plan.get("ease_out", 0.0)))

        for s in range(1, 25):
            if s < start_stage:
                alpha = 0.0
            elif s > end_stage:
                alpha = 1.0
            else:
                raw_alpha = (s - start_stage) / max(1, end_stage - start_stage)
                # Apply easing
                smooth = raw_alpha * raw_alpha * (3.0 - 2.0 * raw_alpha)
                alpha = (1 - ease_in) * raw_alpha + ease_in * smooth
                alpha = (1 - ease_out) * alpha + ease_out * smooth
                alpha = max(0.0, min(1.0, alpha))

            # SLERP for rotation
            trajectory[s, i, :4] = quaternion_normalize(
                quaternion_slerp(q0, q1, alpha)
            )
            # LERP for translation
            trajectory[s, i, 4:7] = (1.0 - alpha) * t0 + alpha * t1

    return trajectory


def _components_for_completion(
    completion: str,
    prompt: str,
    case_gen: DentalCaseGenerator,
    grader: AlignerGrader,
) -> Optional[Dict[str, float]]:
    """
    Score one completion and return its component breakdown:
      total, accuracy, smoothness, compliance, staging.
    Returns None on parse failure (caller substitutes 0.0).
    """
    try:
        seed = abs(hash(prompt)) % (2**31)
        rng = np.random.default_rng(seed)
        try:
            profile = sample_profile("medium", rng)
            case = case_gen.generate_case_for_profile(profile, seed)
        except Exception:
            case = case_gen.generate_case("medium", seed)

        initial = case["initial_config"]
        target = case["target_config"]
        traj = parse_llm_output_to_trajectory(completion, initial, target)
        actual = apply_force_decay(traj, initial)

        # Component scores (decay-realised)
        accuracy = grader.compute_final_accuracy(actual, target)["final_accuracy"]
        smoothness = grader.compute_smoothness(actual)
        compliance = grader.compute_constraint_compliance(actual)["compliance_score"]
        staging = grader.compute_staging_quality(actual, initial, target)
        total, _ = grader.grade("task_medium", actual, initial, target)

        return {
            "total": float(total),
            "accuracy": float(accuracy),
            "smoothness": float(smoothness),
            "compliance": float(compliance),
            "staging": float(staging),
        }
    except Exception:
        return None


def compute_reward(
    trajectory: np.ndarray,
    initial_config: np.ndarray,
    target_config: np.ndarray,
    task_id: str,
    grader: AlignerGrader,
) -> float:
    """Compute total reward for a trajectory with force decay applied."""
    actual = apply_force_decay(trajectory, initial_config)
    reward, _ = grader.grade(
        task_id=task_id,
        agent_traj=actual,
        initial=initial_config,
        target=target_config,
    )
    return reward


# ---------------------------------------------------------------------------
# GRPO reward functions for TRL — multiple per-component functions matching
# the ShopRLVE / Kube SRE winning pattern (separate curves per component)
# ---------------------------------------------------------------------------
def _normalize_prompts(prompts, n_completions):
    if isinstance(prompts, str):
        return [prompts] * n_completions
    if not prompts:
        return [""] * n_completions
    if len(prompts) == 1 and n_completions > 1:
        return prompts * n_completions
    return prompts


def make_reward_fns(case_gen: DentalCaseGenerator, grader: AlignerGrader):
    """
    Build a list of reward functions for GRPOTrainer:
      reward_total, reward_compliance, reward_staging.

    Returning multiple functions makes per-component curves visible in
    wandb (matches ShopRLVE / Kube SRE / Zero Shot Cancer pattern).
    """

    def _scored_batch(completions, prompts):
        prompts = _normalize_prompts(prompts, len(completions))
        return [
            _components_for_completion(c, prompts[i], case_gen, grader)
            for i, c in enumerate(completions)
        ]

    def reward_total(completions: list[str], **kw) -> list[float]:
        prompts = kw.get("prompts", kw.get("prompt", []))
        scores = _scored_batch(completions, prompts)
        return [s["total"] if s else 0.0 for s in scores]

    def reward_compliance(completions: list[str], **kw) -> list[float]:
        prompts = kw.get("prompts", kw.get("prompt", []))
        scores = _scored_batch(completions, prompts)
        return [s["compliance"] if s else 0.0 for s in scores]

    def reward_staging(completions: list[str], **kw) -> list[float]:
        prompts = kw.get("prompts", kw.get("prompt", []))
        scores = _scored_batch(completions, prompts)
        return [s["staging"] if s else 0.0 for s in scores]

    return [reward_total, reward_compliance, reward_staging]


def make_reward_fn(case_gen: DentalCaseGenerator, grader: AlignerGrader):
    """Backwards-compatible single-reward entry point (returns total only)."""
    return make_reward_fns(case_gen, grader)[0]


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
def generate_training_prompts(
    n_prompts: int,
    case_gen: DentalCaseGenerator,
    difficulties: list[str] = ["easy", "medium", "hard"],
) -> list[dict]:
    """Generate training prompts from random dental cases."""
    prompts = []
    for i in range(n_prompts):
        seed = i + 1000
        rng = np.random.default_rng(seed)
        diff = difficulties[i % len(difficulties)]

        try:
            profile = sample_profile(diff, rng)
            case = case_gen.generate_case_for_profile(profile, seed)
        except Exception:
            case = case_gen.generate_case(diff, seed)
            profile = None

        prompt_text = format_observation_prompt(case, profile)
        prompts.append({
            "prompt": prompt_text,
            "seed": seed,
            "difficulty": diff,
        })

    return prompts


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def diagnose_reward_distribution(
    n_prompts: int,
    case_gen: DentalCaseGenerator,
    grader: AlignerGrader,
    perturb_strategies: bool = True,
) -> None:
    """
    Run N synthetic completions through the reward function and print
    distribution stats per component. Used to verify that GRPO will see
    enough reward variance for the group-relative advantage to be useful.

    Synthetic completions are generated by varying tooth_plan parameters
    (start_stage, end_stage, ease_in, ease_out) so the parser produces
    different trajectories — this approximates LLM completion diversity.
    """
    rng = np.random.default_rng(0)
    training_data = generate_training_prompts(n_prompts, case_gen)

    components = {"total": [], "accuracy": [], "smoothness": [], "compliance": [], "staging": []}
    failures = 0

    for td in training_data:
        prompt = td["prompt"]
        # Generate a "completion" with randomised tooth_plans
        if perturb_strategies:
            tooth_plans = []
            for tid in TOOTH_IDS:
                start = int(rng.integers(1, 12))
                end = int(rng.integers(start + 1, 25))
                ease_in = float(rng.uniform(0.0, 1.0))
                ease_out = float(rng.uniform(0.0, 1.0))
                tooth_plans.append({
                    "tooth_id": tid, "start_stage": start, "end_stage": end,
                    "ease_in": ease_in, "ease_out": ease_out,
                })
            completion = json.dumps({"tooth_plans": tooth_plans})
        else:
            completion = json.dumps({"tooth_plans": [
                {"tooth_id": tid, "start_stage": 1, "end_stage": 24, "ease_in": 0, "ease_out": 0}
                for tid in TOOTH_IDS
            ]})

        scored = _components_for_completion(completion, prompt, case_gen, grader)
        if scored is None:
            failures += 1
            continue
        for k, v in scored.items():
            components[k].append(v)

    print(f"\n[diagnose] {len(training_data)} prompts, {failures} failures")
    print(f"  {'component':<14} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
    for k, vals in components.items():
        if not vals:
            continue
        a = np.array(vals)
        print(f"  {k:<14} {a.mean():>8.4f} {a.std():>8.4f} {a.min():>8.4f} {a.max():>8.4f}")
    if components["total"]:
        var = np.var(components["total"])
        print(f"\n  Var(total) = {var:.6f}")
        if var < 1e-4:
            print("  WARNING: reward variance is very low — GRPO advantages will be near-zero.")
        else:
            print("  OK: reward variance is sufficient for GRPO group-relative advantages.")


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for Dental Aligner Planning")
    parser.add_argument("--steps", type=int, default=50, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per step")
    parser.add_argument("--num-generations", type=int, default=4, help="Completions per prompt for GRPO")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--model", type=str, default=MODEL_ID, help="Model ID")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Checkpoint directory")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--max-length", type=int, default=1024, help="Max generation length")
    parser.add_argument("--n-prompts", type=int, default=200, help="Number of training prompts to generate")
    parser.add_argument("--use-unsloth", action="store_true", default=True, help="Use Unsloth 4-bit QLoRA")
    parser.add_argument("--no-unsloth", dest="use_unsloth", action="store_false",
                        help="Skip Unsloth (use plain HF transformers — needs GPU but fewer deps)")
    parser.add_argument("--diagnose", type=int, default=0,
                        help="Run reward-distribution diagnostic over N prompts and exit (no training)")
    parser.add_argument("--single-reward", action="store_true",
                        help="Use a single total-reward function (default: 3 per-component functions)")
    args = parser.parse_args()

    # Diagnose mode runs the reward function over many synthetic completions
    # and reports the distribution — useful before training to confirm variance.
    if args.diagnose > 0:
        case_gen = DentalCaseGenerator()
        grader = AlignerGrader()
        diagnose_reward_distribution(args.diagnose, case_gen, grader)
        return

    print(f"[GRPO] Training config:")
    print(f"  Model: {args.model}")
    print(f"  Steps: {args.steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Generations/prompt: {args.num_generations}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Unsloth QLoRA: {args.use_unsloth}")

    # --- Environment setup ---
    case_gen = DentalCaseGenerator()
    grader = AlignerGrader()

    # --- Generate training dataset ---
    print(f"\n[GRPO] Generating {args.n_prompts} training prompts...")
    training_data = generate_training_prompts(args.n_prompts, case_gen)
    print(f"  Generated {len(training_data)} prompts across easy/medium/hard")

    # --- Compute SLERP baseline score for reference ---
    baseline_scores = []
    for td in training_data[:10]:
        seed = td["seed"]
        rng = np.random.default_rng(seed)
        diff = td["difficulty"]
        try:
            profile = sample_profile(diff, rng)
            case = case_gen.generate_case_for_profile(profile, seed)
        except Exception:
            case = case_gen.generate_case(diff, seed)
        baseline = case["baseline_trajectory"]
        actual = apply_force_decay(baseline, case["initial_config"])
        r, _ = grader.grade(f"task_{diff}", actual, case["initial_config"], case["target_config"])
        baseline_scores.append(r)
    print(f"  SLERP baseline (with decay): {np.mean(baseline_scores):.4f} ± {np.std(baseline_scores):.4f}")

    # --- Model loading ---
    print(f"\n[GRPO] Loading model: {args.model}")

    try:
        if args.use_unsloth:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.resume or args.model,
                max_seq_length=2048,
                dtype=None,  # auto
                load_in_4bit=True,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
            )
            print("  Loaded with Unsloth 4-bit QLoRA")
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.resume or args.model)
            model = AutoModelForCausalLM.from_pretrained(
                args.resume or args.model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            print("  Loaded with standard transformers")
    except ImportError as e:
        print(f"  WARNING: {e}")
        print("  Install with: pip install unsloth trl")
        print("  Falling back to dry-run mode (no actual training)")
        model = None
        tokenizer = None

    if model is None:
        print("\n[GRPO] DRY RUN — testing reward functions only")
        reward_fns = make_reward_fns(case_gen, grader)
        # Test with two SLERP-like and two random outputs to see variance
        rng = np.random.default_rng(0)
        completions = []
        for _ in range(2):
            completions.append(json.dumps({"tooth_plans": [
                {"tooth_id": tid, "start_stage": 1, "end_stage": 24, "ease_in": 0.0, "ease_out": 0.0}
                for tid in TOOTH_IDS
            ]}))
        for _ in range(2):
            completions.append(json.dumps({"tooth_plans": [
                {"tooth_id": tid,
                 "start_stage": int(rng.integers(1, 12)),
                 "end_stage": int(rng.integers(13, 25)),
                 "ease_in": float(rng.uniform(0, 1)),
                 "ease_out": float(rng.uniform(0, 1))}
                for tid in TOOTH_IDS
            ]}))
        test_prompt = training_data[0]["prompt"]
        prompts = [test_prompt] * len(completions)
        for fn in reward_fns:
            rewards = fn(completions, prompts=prompts)
            print(f"  {fn.__name__}: {[round(r, 4) for r in rewards]}")
        print("\n[GRPO] Dry run complete. Install unsloth/trl on a GPU for actual training.")
        return

    # --- TRL GRPO Training ---
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset

    # Build HF dataset from prompts
    dataset = Dataset.from_list([
        {"prompt": td["prompt"]} for td in training_data
    ])

    # GRPO config
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_length,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=min(10, args.steps // 10),
        logging_steps=1,
        save_steps=max(10, args.steps // 5),
        report_to="wandb" if args.wandb else "none",
        run_name=f"orthorl-grpo-{args.steps}steps" if args.wandb else None,
        bf16=True,
        gradient_accumulation_steps=1,
        seed=42,
    )

    if args.single_reward:
        reward_funcs = make_reward_fn(case_gen, grader)
    else:
        # ShopRLVE / Kube SRE / Zero Shot Cancer all used multiple reward
        # functions so wandb shows per-component curves. We do the same.
        reward_funcs = make_reward_fns(case_gen, grader)

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
    )

    print(f"\n[GRPO] Starting training for {args.steps} steps...")
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    print(f"\n[GRPO] Training complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save final model
    final_path = os.path.join(args.output_dir, "final")
    trainer.save_model(final_path)
    print(f"[GRPO] Model saved to {final_path}")

    # --- Post-training evaluation ---
    print("\n[GRPO] Post-training evaluation...")
    # TODO: Generate completions with trained model and compare to baseline


if __name__ == "__main__":
    main()
