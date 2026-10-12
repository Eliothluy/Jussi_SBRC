"""
RSLAQ Policy Evaluation.

Evaluates a trained SAC or DDQN checkpoint over N episodes and compares
against fixed baselines.

Usage:
    python rslaq_eval_policy.py \
        --algo sac \
        --checkpoint results/sac_best.pt \
        --scenario normal \
        --episodes 5 \
        --output results/eval
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import csv
import json
import uuid
import random
import numpy as np
import torch

from environments.rslaq_env import RslaqEnv
from environments.rslaq_action_spaces import build_discrete_action_table

DEFAULT_NS3_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "ns-3-dev"
)
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "results", "eval")

SCENARIO_UE_PROFILES = {
    "low_traffic": (2, 2, 6),
    "normal": (5, 5, 10),
    "congestion": (15, 10, 35),
    "stressed": (8, 12, 20),
    "insufficient_resources": (20, 10, 40),
}


def apply_ue_profile(config, scenario, args):
    cli_profile = (args.embbUes, args.urllcUes, args.mtcUes)
    if all(v > 0 for v in cli_profile):
        profile = cli_profile
    elif any(v > 0 for v in cli_profile):
        raise ValueError("Provide --embbUes, --urllcUes and --mtcUes together, or leave all as 0")
    else:
        profile = SCENARIO_UE_PROFILES.get(scenario, SCENARIO_UE_PROFILES["normal"])

    config["embbUes"] = [int(profile[0])]
    config["urllcUes"] = [int(profile[1])]
    config["mtcUes"] = [int(profile[2])]
    return profile


def ue_profile_from_config(config):
    result = []
    for k in ("embbUes", "urllcUes", "mtcUes"):
        v = config[k]
        if isinstance(v, (list, tuple)):
            result.append(int(v[0]))
        else:
            result.append(int(v))
    return tuple(result)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FixedBaselineAgent:
    """Agent that always returns a fixed PRB allocation."""

    def __init__(self, weights, action_mode="continuous"):
        self.weights = np.array(weights, dtype=np.float32)
        self.action_mode = action_mode

    def act(self, state):
        if self.action_mode == "continuous":
            # Return raw action that after softmax + P_STA yields the desired weights
            # This is approximate; for evaluation we directly send the PRB pct
            return self.weights
        else:
            # For discrete, just return index 0 (will be overridden by eval script)
            return 0


def parse_weights(weights: str) -> np.ndarray:
    values = np.array([float(x.strip()) for x in weights.split(",")], dtype=np.float64)
    if values.shape[0] != 3:
        raise ValueError("--p_sta_weights must contain exactly 3 comma-separated values")
    if values.sum() <= 0:
        raise ValueError("--p_sta_weights must sum to a positive value")
    return values / values.sum()


def load_sac_agent(checkpoint_path, state_shape):
    from examples.rslaq_train_sac import SACActor

    actor = SACActor(state_shape, action_dim=3).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    def act(state):
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
            a, _ = actor.sample(s)
        return a.cpu().numpy().flatten()

    return act


def load_ddqn_agent(checkpoint_path, state_shape, action_size):
    # Robust import: works whether launched from ns-o-ran-gym/ or elsewhere
    import importlib.util
    _ddqn_path = os.path.join(os.path.dirname(__file__), "rslaq_train_ddqn.py")
    _spec = importlib.util.spec_from_file_location("rslaq_train_ddqn", _ddqn_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    QNetwork = _mod.QNetwork

    net = QNetwork(state_shape, action_size).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["online"])
    net.eval()

    def act(state):
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
            q = net(s)
        return q.argmax(dim=1).item()

    return act


def evaluate_agent(
    agent_fn,
    env_config,
    ns3_path,
    output_folder,
    episodes,
    action_mode,
    observation_mode,
    max_steps,
    run_id,
    agent_name,
    include_scheduler=False,
    enable_step_logging=True,
    step_log_file="step_metrics.csv",
    p_sta_weights=None,
    p_sta_static_fraction=0.5,
    sla_config=None,
):
    env = RslaqEnv(
        ns3_path=os.path.abspath(ns3_path),
        scenario_configuration=env_config.copy(),
        output_folder=output_folder,
        optimized=False,
        action_mode=action_mode,
        observation_mode=observation_mode,
        max_steps=max_steps if max_steps != "auto" else None,
        apply_p_sta=True,  # P_STA applied in Python, ns-3 receives final values
        p_sta_weights=p_sta_weights,
        p_sta_static_fraction=p_sta_static_fraction,
        sla_config=sla_config or {},
        include_scheduler=include_scheduler,
        enable_step_logging=enable_step_logging,
        step_log_file=step_log_file,
    )

    rows = []
    for ep in range(episodes):
        obs, info = env.reset()
        state = obs.copy()
        ep_reward = 0.0
        ep_outages = 0
        ep_soft = 0
        actions_log = []

        for step in range(env.max_steps):
            action = agent_fn(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = next_obs.copy()
            done = terminated or truncated

            ep_reward += reward
            actions_log.append(info.get("action_info", {}).get("prb_pct", [0, 0, 0]))

            if info.get("outage_flags"):
                if any(info["outage_flags"].values()):
                    ep_outages += 1
            if info.get("soft_flags"):
                if any(info["soft_flags"].values()):
                    ep_soft += 1

            state = next_state
            if done:
                break

        avg_action = np.mean(actions_log, axis=0) if actions_log else [0, 0, 0]
        final_embb = info.get("slice_0_eMBB", {}).get("throughput_mbps", 0.0)
        final_urllc = info.get("slice_1_URLLC", {}).get("plr_mean", 0.0)
        final_mtc = info.get("slice_2_MTC", {}).get("dLostPackets_sum", 0.0)

        rows.append(
            {
                "run_id": run_id,
                "agent": agent_name,
                "episode": ep + 1,
                "scenario": env.scenario_name,
                "total_reward": ep_reward,
                "steps": step + 1,
                "outages": ep_outages,
                "soft_violations": ep_soft,
                "final_embb_thr": final_embb,
                "final_urllc_plr": final_urllc,
                "final_mtc_lost": final_mtc,
                "avg_action_embb": avg_action[0],
                "avg_action_urllc": avg_action[1],
                "avg_action_mtc": avg_action[2],
                "embb_ues": ue_profile_from_config(env.scenario_configuration)[0],
                "urllc_ues": ue_profile_from_config(env.scenario_configuration)[1],
                "mtc_ues": ue_profile_from_config(env.scenario_configuration)[2],
            }
        )

    env.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description="RSLAQ Policy Evaluation")
    parser.add_argument("--algo", type=str, required=True, choices=["sac", "ddqn"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--scenario", type=str, default="normal")
    parser.add_argument("--embbUes", type=int, default=0,
                        help="Override eMBB UE count; 0 uses the scenario profile")
    parser.add_argument("--urllcUes", type=int, default=0,
                        help="Override URLLC UE count; 0 uses the scenario profile")
    parser.add_argument("--mtcUes", type=int, default=0,
                        help="Override MTC UE count; 0 uses the scenario profile")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval_seeds", type=str, default="",
                        help="Comma-separated seeds for multi-seed evaluation (e.g., '1,2,3,4,5'). Overrides --seed if set.")
    parser.add_argument("--ns3_path", type=str, default=DEFAULT_NS3_PATH)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--simTime", type=float, default=4.0)
    parser.add_argument("--appStart", type=float, default=0.5)
    parser.add_argument("--periodMs", type=int, default=10)
    parser.add_argument("--max_steps", type=str, default="auto")
    parser.add_argument("--observation_mode", type=str, default="paper")
    parser.set_defaults(include_scheduler=True)
    parser.add_argument("--no-include-scheduler", action="store_false", dest="include_scheduler",
                        help="Disable scheduler selection for DDQN evaluation")
    parser.set_defaults(enable_step_logging=True)
    parser.add_argument("--no-step-logging", action="store_false", dest="enable_step_logging",
                        help="Disable per-step step_metrics.csv logging")
    parser.add_argument("--step_log_file", type=str, default="step_metrics.csv",
                        help="Per-simulation step metrics filename")
    parser.add_argument("--p_sta_static_fraction", type=float, default=0.5,
                        help="Static fraction in P_STA decomposition")
    parser.add_argument("--p_sta_weights", type=str, default="0.3333,0.4000,0.2667",
                        help="Comma-separated P_STA static weights")
    parser.add_argument("--reward_alpha", type=float, default=0.3333,
                        help="Reward weight for eMBB")
    parser.add_argument("--reward_beta", type=float, default=0.4000,
                        help="Reward weight for URLLC")
    parser.add_argument("--reward_gamma", type=float, default=0.2667,
                        help="Reward weight for MTC")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)
    run_id = str(uuid.uuid4())[:8]

    env_config = {
        "simTime": [args.simTime],
        "appStart": [args.appStart],
        "scenario": [args.scenario],
        "seed": [args.seed],
        "periodMs": [args.periodMs],
    }
    ue_profile = apply_ue_profile(env_config, args.scenario, args)

    action_mode = "continuous" if args.algo == "sac" else "discrete"
    include_scheduler = bool(args.include_scheduler and args.algo == "ddqn")
    p_sta_weights = parse_weights(args.p_sta_weights)
    sla_config = {
        "alpha": args.reward_alpha,
        "beta": args.reward_beta,
        "gamma": args.reward_gamma,
    }

    # Load agent
    if args.algo == "sac":
        # Need state_shape; create temporary env to get it
        tmp_env = RslaqEnv(
            ns3_path=os.path.abspath(args.ns3_path),
            scenario_configuration=env_config.copy(),
            output_folder=args.output,
            optimized=False,
            action_mode=action_mode,
            observation_mode=args.observation_mode,
            max_steps=1,
            apply_p_sta=True,
            p_sta_weights=p_sta_weights,
            p_sta_static_fraction=args.p_sta_static_fraction,
            include_scheduler=False,
            enable_step_logging=False,
        )
        state_shape = tmp_env.observation_space.shape
        tmp_env.close()
        agent_fn = load_sac_agent(args.checkpoint, state_shape)
    else:
        tmp_env = RslaqEnv(
            ns3_path=os.path.abspath(args.ns3_path),
            scenario_configuration=env_config.copy(),
            output_folder=args.output,
            optimized=False,
            action_mode=action_mode,
            observation_mode=args.observation_mode,
            max_steps=1,
            apply_p_sta=True,
            p_sta_weights=p_sta_weights,
            p_sta_static_fraction=args.p_sta_static_fraction,
            include_scheduler=include_scheduler,
            enable_step_logging=False,
        )
        state_shape = tmp_env.observation_space.shape
        action_size = len(build_discrete_action_table(step=0.1, include_scheduler=include_scheduler))
        tmp_env.close()
        agent_fn = load_ddqn_agent(args.checkpoint, state_shape, action_size)

    all_rows = []

    # Determine seeds for evaluation
    if args.eval_seeds:
        eval_seeds = [int(s.strip()) for s in args.eval_seeds.split(",")]
        print(f"Multi-seed evaluation with seeds: {eval_seeds}")
    else:
        eval_seeds = [args.seed]

    # Evaluate trained agent across all seeds
    for seed in eval_seeds:
        print(f"Evaluating {args.algo.upper()} on {args.scenario} (seed={seed}) ...")
        seed_config = env_config.copy()
        seed_config["seed"] = [seed]
        rows = evaluate_agent(
            agent_fn,
            seed_config,
            args.ns3_path,
            args.output,
            args.episodes,
            action_mode,
            args.observation_mode,
            args.max_steps,
            run_id,
            f"{args.algo}_s{seed}",
            include_scheduler=include_scheduler,
            enable_step_logging=args.enable_step_logging,
            step_log_file=args.step_log_file,
            p_sta_weights=p_sta_weights,
            p_sta_static_fraction=args.p_sta_static_fraction,
            sla_config=sla_config,
        )
        all_rows.extend(rows)

    # Baselines (only for continuous/resource mode)
    baselines = {
        "fixed_equal": [33.33, 33.33, 33.34],
        "embb_biased": [50.0, 30.0, 20.0],
        "urllc_biased": [20.0, 50.0, 30.0],
        "mtc_biased": [20.0, 30.0, 50.0],
    }

    for name, weights in baselines.items():
        print(f"Evaluating baseline {name} ...")
        baseline_agent = FixedBaselineAgent(weights, action_mode="continuous")
        for seed in eval_seeds:
            seed_config = env_config.copy()
            seed_config["seed"] = [seed]
            rows = evaluate_agent(
                baseline_agent.act,
                seed_config,
                args.ns3_path,
                args.output,
                args.episodes,
                "continuous",
                args.observation_mode,
                args.max_steps,
                run_id,
                f"{name}_s{seed}",
                include_scheduler=False,
                enable_step_logging=args.enable_step_logging,
                step_log_file=args.step_log_file,
                p_sta_weights=p_sta_weights,
                p_sta_static_fraction=args.p_sta_static_fraction,
                sla_config=sla_config,
            )
            all_rows.extend(rows)

    # Save CSV
    csv_path = os.path.join(args.output, f"eval_{args.algo}_{args.scenario}_{run_id}.csv")
    if all_rows:
        keys = all_rows[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Results saved to {csv_path}")

    summary = {
        "run_id": run_id,
        "algo": args.algo,
        "checkpoint": args.checkpoint,
        "scenario": args.scenario,
        "episodes": args.episodes,
        "seed": args.seed,
        "num_ues_per_slice": {
            "eMBB": ue_profile[0],
            "URLLC": ue_profile[1],
            "MTC": ue_profile[2],
        },
        "num_ues": sum(ue_profile),
    }
    with open(os.path.join(args.output, f"eval_{run_id}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
