#!/usr/bin/env python3
"""Extract the greedy policy weights from trained DDQN checkpoints.

For each ddqn_paper_<scenario>_seed<N>/ directory, loads ddqn_best.pt (or
ddqn_final.pt), runs the policy greedily (epsilon=0) for a small number of
evaluation episodes, and records the mean action (PRB weights) actually
applied. The output CSV feeds the standalone re-run that produces a v2-Path-C
score comparable to the meta-heuristics.

Usage:
    python3 examples/extract_ddqn_weights.py \
        --ddqn-root results_controlled/heuristics_metaheuristics/20260811_rslaq_ddqn_paper/rslaq_ddqn_paper \
        --output ddqn_extracted_weights.csv \
        --eval-episodes 3

The script reuses QNetwork/RslaqEnv from the existing codebase without
retraining: it instantiates the env in eval mode (no learning), loads the
checkpoint, and records the action_info['prb_pct'] returned by env.step().
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

# Make the ns-o-ran-gym src importable when run from examples/
HERE = os.path.dirname(os.path.abspath(__file__))
GYM_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(GYM_DIR, "src"))
sys.path.insert(0, HERE)

# Reuse the existing implementation
from rslaq_train_ddqn import QNetwork, build_discrete_action_table  # noqa: E402
from environments.rslaq_env import RslaqEnv  # noqa: E402

import torch  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIR_RE = re.compile(r"ddqn_paper_(.*)_seed(\d+)$")


def load_qnetwork(ckpt_path: str, state_shape, num_actions: int):
    net = QNetwork(state_shape, num_actions).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("online", ckpt)
    # QNetwork checkpoints may store raw state_dict or a wrapper dict
    if isinstance(state, dict) and "online" in state:
        state = state["online"]
    net.load_state_dict(state)
    net.eval()
    return net


def greedy_act(net, state):
    with torch.no_grad():
        s = torch.FloatTensor(np.asarray(state)).unsqueeze(0).unsqueeze(0).to(DEVICE)
        q = net(s)
    return int(q.argmax(dim=1).item())


def build_eval_env(scenario, seed, ns3_path, output_folder, include_scheduler,
                   p_sta_weights, p_sta_static_fraction):
    """Build a RslaqEnv configured for a single (scenario, seed) eval run."""
    action_table = build_discrete_action_table(
        step=0.1, include_scheduler=include_scheduler
    )
    num_actions = len(action_table)
    # Mirror the config used during training (see rslaq_train_ddqn.train_ddqn).
    scenario_configuration = {
        "scenario": [scenario],
        "seed": [seed],
        "seed_cycle": [1],            # fixed seed during eval
        "simTime": [5.0],
        "appStart": [0.5],
        "periodMs": [10],
    }
    sla_config = {
        "reward_weights": [0.3333, 0.4000, 0.2667],
        "p_sta_weights": p_sta_weights,
        "p_sta_static_fraction": p_sta_static_fraction,
    }
    env = RslaqEnv(
        ns3_path=os.path.abspath(ns3_path),
        scenario_configuration=scenario_configuration,
        output_folder=output_folder,
        optimized=False,
        action_mode="discrete",
        observation_mode="paper",
        max_steps=100,
        apply_p_sta=True,
        p_sta_weights=p_sta_weights,
        p_sta_static_fraction=p_sta_static_fraction,
        sla_config=sla_config,
        include_scheduler=include_scheduler,
        enable_step_logging=False,
    )
    return env, num_actions


def evaluate_run(net, env, num_actions, episodes, ntsr):
    """Run the greedy policy and collect mean PRB weights + paper-reward."""
    prb_accum = np.zeros(3)
    rewards = []
    steps_total = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        state = obs.copy()
        ep_prb = np.zeros(3)
        ep_steps = 0
        ep_reward = 0.0
        for _ in range(ntsr):
            a = greedy_act(net, state)
            next_obs, reward, terminated, truncated, info = env.step(a)
            ep_reward += reward
            ai = info.get("action_info", {})
            prb = ai.get("prb_pct")
            if prb is not None:
                ep_prb = np.array(prb, dtype=float)
            state = next_obs.copy()
            ep_steps += 1
            if terminated or truncated:
                break
        prb_accum += ep_prb
        rewards.append(ep_reward)
        steps_total += ep_steps
    n = max(episodes, 1)
    return prb_accum / n, float(np.mean(rewards)) if rewards else 0.0, steps_total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ddqn-root", required=True,
                    help="Directory containing ddqn_paper_<sc>_seed<N>/ subdirs")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--ns3-dir", default=os.path.join(GYM_DIR, "..", "ns-3-dev"),
                    help="Path to ns-3-dev (for the rslaq-sim binary)")
    ap.add_argument("--eval-episodes", type=int, default=3,
                    help="Greedy episodes to average per (scenario, seed)")
    ap.add_argument("--seeds", default="",
                    help="Comma-separated seed filter (e.g. '1,2,3,4,5'); "
                         "empty = all ddqn_paper_*_seed* dirs")
    ap.add_argument("--scenarios", default="",
                    help="Comma-separated scenario filter (e.g. 'congestion'); "
                         "empty = all scenarios")
    ap.add_argument("--checkpoint", choices=["best", "final"], default="best",
                    help="Which checkpoint to evaluate (default: best)")
    ap.add_argument("--include-scheduler", action="store_true", default=True,
                    help="Action table includes scheduler (paper: 198 actions)")
    ap.add_argument("--p-sta-weights", default="0.3333,0.4000,0.2667")
    ap.add_argument("--p-sta-static-fraction", type=float, default=0.5)
    args = ap.parse_args()

    p_sta_weights = [float(x) for x in args.p_sta_weights.split(",")]
    ns3_path = os.path.abspath(args.ns3_dir)

    runs = sorted(d for d in os.listdir(args.ddqn_root) if DIR_RE.match(d))
    if args.seeds:
        wanted = {int(s) for s in args.seeds.split(",") if s.strip()}
        runs = [d for d in runs if int(DIR_RE.match(d).group(2)) in wanted]
        print(f"[FILTER] --seeds={sorted(wanted)} -> {len(runs)} runs selected")
    if args.scenarios:
        wanted_scn = {s.strip() for s in args.scenarios.split(",") if s.strip()}
        runs = [d for d in runs if DIR_RE.match(d).group(1) in wanted_scn]
        print(f"[FILTER] --scenarios={sorted(wanted_scn)} -> {len(runs)} runs selected")
    if not runs:
        print(f"[WARN] No ddqn_paper_*_seed* directories in {args.ddqn_root}",
              file=sys.stderr)
        return 1

    out_rows = []
    for run in runs:
        m = DIR_RE.match(run)
        scenario, seed_str = m.group(1), m.group(2)
        seed = int(seed_str)
        run_dir = os.path.join(args.ddqn_root, run)

        # Read training summary for final_avg / best_avg context
        summary_path = os.path.join(run_dir, "ddqn_summary.json")
        training_info = {}
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                training_info = json.load(f)

        ckpt_name = f"ddqn_{args.checkpoint}.pt"
        ckpt_path = os.path.join(run_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] {run}: {ckpt_name} not found", file=sys.stderr)
            continue

        # Build env + net. state_shape matches the paper observation (4,4).
        eval_out = os.path.join(run_dir, "extract_eval")
        os.makedirs(eval_out, exist_ok=True)
        env, num_actions = build_eval_env(
            scenario, seed, ns3_path, eval_out,
            args.include_scheduler, p_sta_weights, args.p_sta_static_fraction,
        )
        state_shape = (4, 4)
        net = load_qnetwork(ckpt_path, state_shape, num_actions)

        print(f"[EVAL] {run} checkpoint={ckpt_name} episodes={args.eval_episodes}")
        mean_prb, mean_reward, steps = evaluate_run(
            net, env, num_actions, args.eval_episodes, ntsr=100
        )
        # Normalize PRB percentages to a simplex (they should already sum ~100)
        total = mean_prb.sum()
        w = (mean_prb / total) if total > 0 else np.array([1/3, 1/3, 1/3])

        out_rows.append({
            "scenario": scenario,
            "seed": seed,
            "checkpoint": ckpt_name,
            "w_embb": round(float(w[0]), 6),
            "w_urllc": round(float(w[1]), 6),
            "w_mtc": round(float(w[2]), 6),
            "mean_prb_embb": round(float(mean_prb[0]), 4),
            "mean_prb_urllc": round(float(mean_prb[1]), 4),
            "mean_prb_mtc": round(float(mean_prb[2]), 4),
            "mean_paper_reward": round(mean_reward, 6),
            "eval_steps": steps,
            "best_avg_reward_training": training_info.get("best_avg"),
            "final_avg_reward_training": training_info.get("final_avg_100"),
        })
        env.close()
        print(f"   -> w=[{w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f}] "
              f"reward={mean_reward:.4f}")

    # Write CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        fieldnames = [
            "scenario", "seed", "checkpoint",
            "w_embb", "w_urllc", "w_mtc",
            "mean_prb_embb", "mean_prb_urllc", "mean_prb_mtc",
            "mean_paper_reward", "eval_steps",
            "best_avg_reward_training", "final_avg_reward_training",
        ]
        w_csv = csv.DictWriter(f, fieldnames=fieldnames)
        w_csv.writeheader()
        w_csv.writerows(out_rows)
    print(f"\n[OK] {len(out_rows)} runs -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
