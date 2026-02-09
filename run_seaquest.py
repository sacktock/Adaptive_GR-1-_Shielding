import os
import torch
import numpy as np
import pandas as pd

import gymnasium as gym
from env.seaquest import Seaquest, ClipRewardWrapper
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from wrappers.adaptive_shield import AdaptiveShieldWrapper
from wrappers.seaquest_shields import Naive_Shield, Static_Shield, Repaired_Shield, Adaptive_Shield
from wrappers.shield import ShieldWrapper
from spec_repair.config import PROJECT_PATH

from helpers import linear_schedule, EpisodeAccumulator, evaluate_policy, mean_se

import argparse

PATH_TO_SPEC = os.path.join(PROJECT_PATH, "tests/shield_test/submarine_boolean_64_92.spectra")

ENV_KEYS=[
    "diver_at_depth1",
    "diver_at_depth2",
    "diver_at_depth3",
    "diver_at_depth4",
    "reset_flag",
    "operational",
    "oxygen0",
    "oxygen1",
    "oxygen2",
    "oxygen3",
    "oxygen4",
    "oxygen5",
    "oxygen6",
    "depth0",
    "depth1",
    "depth2",
    "depth3",
    "depth4"
]

def sys_abstr(action):
    return {
        "up": "true" if action in [2,6,7,10,14,15] else "false",
        "down": "true" if action in [5,8,9,13,16,17] else "false"
    }

def act_abstr(outputs, varibs, action):
    if outputs ["up"] == "false" and outputs ["down"] == "false":
        return 0
    if outputs["up"] == "true" and outputs ["down"] == "false":
        return 2
    if outputs ["up"] == "false" and outputs["down"] == "true":
        return 5
    return action 

def make_env(seed=0, max_steps=108000, clip_reward=True, shield_impl="none"):
    """
    Seaquest (expected) -> ShieldWrapper -> Monitor -> (optional) ClipReward
    """
    def _init():
        env = Seaquest(
            repeat=4,
            size=(84,84),
            gray=True,
            noops=0,
            lives='unused', 
            sticky=False,
            initial_oxygen_depletion_rate=1,
            unexpected_violation=False,
            max_episode_steps=max_steps,
        )
        if shield_impl == "adaptive-ilasp":
            env = AdaptiveShieldWrapper(
                env,
                PATH_TO_SPEC,
                env_keys=ENV_KEYS,  # env vars the wrapper expects
                sys_abstr=sys_abstr,
                act_abstr=act_abstr,
            )
        if shield_impl == "naive":
            env = ShieldWrapper(env, Naive_Shield())
        if shield_impl == "static":
            env = ShieldWrapper(env, Static_Shield())
        if shield_impl == "repaired":
            env = ShieldWrapper(env, Repaired_Shield())
        if shield_impl == "adaptive-python":
            env = ShieldWrapper(env, Adaptive_Shield())
        env = Monitor(env, info_keywords=("is_success",))
        if clip_reward:
            env = ClipRewardWrapper(env)
        env.reset(seed=seed)
        return env
    return _init

def make_eval_env(seed=0, max_steps=108000, shield_impl="none"):
    """
    Seaquest (unexpected) -> ShieldWrapper
    """
    def _init():
        env = Seaquest(
            repeat=4,
            size=(84,84),
            gray=True,
            noops=0,
            lives='unused', 
            sticky=False,
            initial_oxygen_depletion_rate=1,
            unexpected_violation=True,
            max_episode_steps=max_steps,
        )
        if shield_impl == "adaptive-ilasp":
            env = AdaptiveShieldWrapper(
                env,
                PATH_TO_SPEC,
                env_keys=ENV_KEYS,  # env vars the wrapper expects
                sys_abstr=sys_abstr,
                act_abstr=act_abstr,
            )
        if shield_impl == "naive":
            env = ShieldWrapper(env, Naive_Shield())
        if shield_impl == "static":
            env = ShieldWrapper(env, Static_Shield())
        if shield_impl == "repaired":
            env = ShieldWrapper(env, Repaired_Shield())
        if shield_impl == "adaptive-python":
            env = ShieldWrapper(env, Adaptive_Shield())
        env.reset(seed=seed)
        return env
    return _init

def one_run(run_seed: int, n_envs: int, total_timesteps: int, n_eval_episodes: int, log_root: str, tb: bool = False, shield_impl: str = "none"):
    run_dir = os.path.join(log_root, f"ppo_seed_{run_seed}")
    os.makedirs(run_dir, exist_ok=True)

    output = {}

    checkpoint_path = os.path.join(run_dir, "chkpt.zip")

    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}, loading...")
        model = PPO.load(checkpoint_path)  
    else:
        print("No checkpoint found, commencing training ...")

        seeds = [run_seed + i for i in range(n_envs)]
        train_env = SubprocVecEnv([make_env(seed=seed, clip_reward=True, shield_impl=shield_impl) for seed in seeds])

        model = PPO(
            policy="CnnPolicy",
            env=train_env,
            n_steps=128,                 
            batch_size=256,              
            n_epochs=3,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=1.0,
            learning_rate=linear_schedule(2.5e-4),
            clip_range=linear_schedule(0.1),
            max_grad_norm=0.5,
            stats_window_size=100,
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

        save_path = os.path.join(run_dir, "chkpt")
        model.save(save_path)
        print(f"Model saved to: {save_path}.zip")

        train_env.close()

    #eval_env = DummyVecEnv([make_env(seed=seed, clip_reward=False, shield_impl=shield_impl)])

    eval_env = DummyVecEnv([make_eval_env(seed=seed+123, shield_impl=shield_impl)])

    output.update(evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes))

    eval_env.close()

    output.update({"seed": run_seed})

    return output

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--runs", type=int, default=5)
    parser.add_argument("-n", "--n-envs", type=int, default=8)
    parser.add_argument("-t", "--timesteps", type=int, default=10_000_000)
    parser.add_argument("-e", "--eval", type=int, default=20)
    parser.add_argument("--logdir", type=str, default="./logdir/seaquest")
    parser.add_argument("--tensorboard", action="store_true", default=False)
    parser.add_argument("--shield", type=str, choices=["none", "naive", "static", "repaired", "adaptive-python", "adaptive-ilasp"], default="adaptive-python")
    args = parser.parse_args()

    runs = args.runs
    n_envs = args.n_envs
    timesteps = args.timesteps
    eval_episodes = args.eval
    log_root = os.path.join(args.logdir, f"ppo_{args.shield}")
    os.makedirs(log_root, exist_ok=True)

    # metrics to log and print
    metrics = [
        "train_reward_mean",
        "train_success_mean",
        "train_override_mean",
        "train_success_1_mean",
        "train_winning_region_mean",
        "train_guarantee_1_mean",
        "eval_reward_mean",
        "eval_reward_std",
        "eval_success_mean",
        "eval_override_mean",
        "eval_success_1_mean",
        "eval_in_winning_region_mean",
        "eval_guarantee_1_mean",
    ]

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
        res = one_run(seed, n_envs, timesteps, eval_episodes, log_root, tb=args.tensorboard, shield_impl=args.shield)
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