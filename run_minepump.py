import os
import torch
import pandas as pd

import gymnasium as gym
from env.minepump import MinePumpEnvV1
from gymnasium.wrappers import TimeLimit, FlattenObservation
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from wrappers.adaptive_shield import AdaptiveShieldWrapper
from wrappers.shield import ShieldWrapper
from wrappers.minepump_shields import Static_Shield_Star, Static_Shield_1, Static_Shield_2
from spec_repair.config import PROJECT_PATH

from helpers import EpisodeAccumulator, evaluate_controller, evaluate_policy, mean_se

import argparse

PATH_TO_SPEC = os.path.join(PROJECT_PATH, "tests/shield_test/minepump_strong.spectra")

def make_env(seed=0, max_steps=2000, shield_impl="none"):
    """
    Minepump (expected) -> AdaptiveShield/Shield -> TimeLimit -> FlattenObservation -> Monitor
    """
    def _init():
        env = MinePumpEnvV1(render_mode=None, unexpected_violation=False)
        if shield_impl == "adaptive":
            env = AdaptiveShieldWrapper(
                env,
                seed,
                PATH_TO_SPEC,
                env_keys=["highwater", "methane"],  # env vars the wrapper expects
                sys_abstr=lambda action: {"pump": "true" if action == 1 else "false"},
                act_abstr=lambda output, varibs, action: 1 if output["pump"] == "true" else 0,
                initiate_spec_repair=False,
            )
        if shield_impl == "static_1":
            env = ShieldWrapper(env, Static_Shield_1())
        if shield_impl == "static_2":
            env = ShieldWrapper(env, Static_Shield_2())
        if shield_impl == "static_star":
            env = ShieldWrapper(env, Static_Shield_Star())
        env = TimeLimit(env, max_episode_steps=max_steps)
        env = FlattenObservation(env)
        env = Monitor(env, info_keywords=("is_success",))
        env.reset(seed=seed)
        return env
    return _init

def make_eval_env(seed=0, max_steps=2000, shield_impl="none"):
    """
    Minepump (unexpected) -> AdaptiveShield/Shield -> TimeLimit -> FlattenObservation
    """
    def _init():
        env = MinePumpEnvV1(render_mode=None, unexpected_violation=True)
        if shield_impl == "adaptive":
            env = AdaptiveShieldWrapper(
                env,
                seed,
                PATH_TO_SPEC,
                env_keys=["highwater", "methane"],  # env vars the wrapper expects
                sys_abstr=lambda action: {"pump": "true" if action == 1 else "false"},
                act_abstr=lambda output, varibs, action: 1 if output["pump"] == "true" else 0,
                initiate_spec_repair=True,
            )
        if shield_impl == "static_1":
            env = ShieldWrapper(env, Static_Shield_1())
        if shield_impl == "static_2":
            env = ShieldWrapper(env, Static_Shield_2())
        if shield_impl == "static_star":
            env = ShieldWrapper(env, Static_Shield_Star())
        env = TimeLimit(env, max_episode_steps=max_steps)
        env = FlattenObservation(env)
        env.reset(seed=seed)
        return env
    return _init

def one_run(run_seed: int, total_timesteps: int, n_eval_episodes: int, log_root: str, tb: bool = False, shield_impl: str = "none"):
    run_dir = os.path.join(log_root, f"dqn_seed_{run_seed}")
    os.makedirs(run_dir, exist_ok=True)

    output = {}

    checkpoint_path = os.path.join(run_dir, "ckhpt.zip")

    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}, loading...")
        model = DQN.load(checkpoint_path)  
    else:
        print("No checkpoint found, commencing training ...")

        train_env = DummyVecEnv([make_env(seed=run_seed, max_steps=2000, shield_impl=shield_impl)])

        model = DQN(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=1e-3,
            buffer_size=100_000,
            learning_starts=10_000,
            batch_size=256,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1_000,
            exploration_fraction=0.2,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            policy_kwargs=dict(net_arch=[64, 64]),
            max_grad_norm=0.5,
            stats_window_size=10,
            tensorboard_log=run_dir if tb else None,
            verbose=1,
            device="auto",
            seed=run_seed,
        )

        print("Torch CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("CUDA device:", torch.cuda.get_device_name(0))
        print("SB3 model device:", model.device)

        cb = EpisodeAccumulator()
        model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=cb)

        output.update(
            cb.get_training_averages()
        )

        save_path = os.path.join(run_dir, "ckhpt")
        model.save(save_path)
        print(f"Model saved to: {save_path}.zip")

        train_env.close()

    eval_env = DummyVecEnv([make_eval_env(seed=run_seed+123, max_steps=2000, shield_impl=shield_impl)])

    output.update(evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes))

    eval_env.close()

    output.update({"seed": run_seed})

    return output

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--runs", type=int, default=10)
    parser.add_argument("-t", "--timesteps", type=int, default=200_000)
    parser.add_argument("-e", "--eval", type=int, default=20)
    parser.add_argument("--logdir", type=str, default="./logdir/minepump")
    parser.add_argument("--tensorboard", action="store_true", default=False)
    parser.add_argument("--shield", type=str, choices=["none", "static_1", "static_2", "static_star", "adaptive"], default="adaptive")
    args = parser.parse_args()

    runs = args.runs
    timesteps = args.timesteps
    eval_episodes = args.eval
    log_root = os.path.join(args.logdir, f"dqn_{args.shield}")
    os.makedirs(log_root, exist_ok=True)

    # metrics to log and print
    metrics = [
        "train_reward_mean",
        "train_success_mean",
        "train_override_mean",
        "train_success_1_mean",
        "train_success_2_mean",
        "train_winning_region_mean",
        "train_guarantee_1_mean",
        "train_guarantee_2_mean",
        "eval_reward_mean",
        "eval_reward_std",
        "eval_success_mean",
        "eval_override_mean",
        "eval_success_1_mean",
        "eval_success_2_mean",
        "eval_in_winning_region_mean",
        "eval_guarantee_1_mean",
        "eval_guarantee_2_mean",
    ]

    """Expected environemnt eval for the static symbolic controllers"""

    env = MinePumpEnvV1(render_mode=None, unexpected_violation=False)
    env = TimeLimit(env, max_episode_steps=2000)
    env.reset(seed=42)

    # static symbolic controller before spec repair (expected env)
    static_controller_1 = lambda obs: 1 if (obs["prev_highwater"] == 1) else 0
    res = evaluate_controller(static_controller_1, env, n_eval_episodes=eval_episodes)
    print(f"\n === [Static Symbolic Contoller 1 (expected)] ===")
    for key in res:
        print(f"    {key}: {res[key]:.2f}")

    env.close()

    env = MinePumpEnvV1(render_mode=None, unexpected_violation=False)
    env = TimeLimit(env, max_episode_steps=2000)
    env.reset(seed=42)

    # static symbolic controller after spec repair (expected env)
    static_controller_2 = lambda obs: 1 if (obs["prev_highwater"] == 1 and obs["prev_methane"] == 0) else 0
    res = evaluate_controller(static_controller_2, env, n_eval_episodes=eval_episodes)
    print(f"\n === [Static Symbolic Contoller 2 (expected)] ===")
    for key in res:
        print(f"    {key}: {res[key]:.2f}")

    env.close()

    """Unexpected environemnt eval for the static symbolic controllers"""

    env = MinePumpEnvV1(render_mode=None, unexpected_violation=True)
    env = TimeLimit(env, max_episode_steps=2000)
    env.reset(seed=42)

    # static symbolic controller before spec repair (unexpected env)
    static_controller_1 = lambda obs: 1 if (obs["prev_highwater"] == 1) else 0
    res = evaluate_controller(static_controller_1, env, n_eval_episodes=eval_episodes)
    print(f"\n === [Static Symbolic Contoller 1 (unexpected)] ===")
    for key in res:
        print(f"    {key}: {res[key]:.2f}")

    env.close()

    env = MinePumpEnvV1(render_mode=None, unexpected_violation=True)
    env = TimeLimit(env, max_episode_steps=2000)
    env.reset(seed=42)

    # static symbolic controller after spec repair (unexpected env)
    static_controller_2 = lambda obs: 1 if (obs["prev_highwater"] == 1 and obs["prev_methane"] == 0) else 0
    res = evaluate_controller(static_controller_2, env, n_eval_episodes=eval_episodes)
    print(f"\n === [Static Symbolic Contoller 2 (unexpected)] ===")
    for key in res:
        print(f"    {key}: {res[key]:.2f}")

    env.close()

    seeds = [i for i in range(runs)]

    results_dir = os.path.join(log_root, "results.csv")

    if os.path.exists(results_dir):
        df = pd.read_csv(results_dir)
        results = df[metrics].to_dict(orient="records")
        print(f"Loaded past results from {results_dir} ...")
    else:
        results = []
        print(f"No past results found, starting fresh runs ...")

    for i, seed in enumerate(seeds, 0):
        print(f"\n=== Run {i+1}/{runs} (seed={seed}) ===")
        res = one_run(seed, timesteps, eval_episodes, log_root, tb=args.tensorboard, shield_impl=args.shield)
        try:
            for key in metrics:
                if key[0:4] == "eval":
                    results[i][key] = res[key]
        except:
            results.append(res)

        # sanity checks
        for key in metrics:
            assert key in results[i].keys(), f"something went wrong: {key} not in results[i].keys()"

        # drop any unnecessary metrics
        results[i] = {k: v for k, v in results[i].items() if k in metrics}

        print(f"  train_reward_mean: {results[i]['train_reward_mean']:.2f}")
        print(f"  train_success_mean: {results[i]['train_success_mean']:.2f}")
        print(f"  eval_reward_mean: {results[i]['eval_reward_mean']:.2f} +/- {results[i]['eval_reward_std']:.2f}")
        print(f"  eval_success_mean: {results[i]['eval_success_mean']:.2f}")
        print(f"  eval_override_mean: {results[i]['eval_override_mean']:.2f}")
        print("\n=== Extra diagnositics ===")
        for key in metrics:
            if key in ['train_reward_mean', 'train_success_mean', 'eval_reward_mean', 'eval_reward_std', 'eval_success_mean', 'eval_override_mean']:
                continue
            else:
                print(f"    {key}: {results[i][key]:.2f}")
        print()


    # write results to csv
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(log_root, "results.csv"), index=False)
    print(f"Per-run details saved to {os.path.join(log_root, 'results.csv')}")

    train_rew = [r["train_reward_mean"] for r in results]
    train_succ = [r["train_success_mean"] for r in results]
    eval_rew = [r["eval_reward_mean"] for r in results]
    eval_succ = [r["eval_success_mean"] for r in results]
    eval_ovr = [r["eval_override_mean"] for r in results]

    m_tr, se_tr = mean_se(train_rew)
    m_ts, se_ts = mean_se(train_succ)
    m_er, se_er = mean_se(eval_rew)
    m_es, se_es = mean_se(eval_succ)
    m_eo, se_eo = mean_se(eval_ovr)

    print("\n=== Summary over runs ===")
    print(f"Training Reward: mean={m_tr:.2f} +/- SE={se_tr:.2f}")
    print(f"Training Success: mean={m_ts:.2f} +/- SE={se_ts:.2f}")
    print(f"Eval Reward: mean={m_er:.2f} +/- SE={se_er:.2f}")
    print(f"Eval Successs: mean={m_es:.2f} +/- SE={se_es:.2f}")
    print(f"Eval Override: mean={m_eo:.2f} +/- SE={se_eo:.2f}")


    