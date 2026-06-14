import os
import torch

import gymnasium as gym
from env.seaquest import Seaquest, ClipRewardWrapper

from typing import Dict, Any

import masa
from masa.algorithms.on_policy import CPO, PPOLag, TRPOLag
from masa.common.wrappers import TimeLimit, ConstraintMonitor, RewardMonitor, ConstraintPersistentGymnasiumWrapper, ConstraintPersistentWrapper
from masa.common.labelled_env import LabelledEnv
from masa.common.label_fn import LabelFn
from masa.common import registry
from masa.common.ltl import *
from masa.common.layers import NatureCNN
from typing import Dict, Any
import numpy as np

import argparse

import jax

jax.config.update("jax_platform_name", "cpu")

class WinningRegionMonitor(ConstraintPersistentWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.in_winning_region = []
        self.guarantee_1 = []

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)

        self.in_winning_region = []
        self.guarantee_1 = []

        if "step" in info["constraint"]:
            self.in_winning_region.append(bool(info["in_winning_region"]))
            self.guarantee_1.append(bool(info["guarantee_1"]))
        
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "step" in info["constraint"]:
            self.in_winning_region.append(bool(info["in_winning_region"]))
            self.guarantee_1.append(bool(info["guarantee_1"]))

        if "episode" in info["constraint"]:
            info["constraint"]["episode"]["guarantee_1"] = np.mean(self.guarantee_1)
            info["constraint"]["episode"]["in_winning_region"] = np.mean(self.in_winning_region)

        return observation, reward, terminated, truncated, info

class DummyLabelledEnv(LabelledEnv):

    def __init__(self, env: gym.Env):
        super().__init__(env, None)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        assert "labels" in info
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        assert "labels" in info
        return obs, reward, terminated, truncated, info

cost_fn = lambda labels: 1.0 if "safe" in labels else 0.0

def make_env(seed=0, max_steps=2000, clip_reward=True):
    env = Seaquest(
            repeat=4,
            size=(84,84),
            gray=True,
            noops=0,
            lives='reset', 
            sticky=False,
            transpose_obs=True,
            initial_oxygen_depletion_rate=1,
            unexpected_violation=False,
            max_episode_steps=max_steps,
        )
    constraint_ctor = registry.get_constraint("CMDP")
    env = DummyLabelledEnv(env)
    constraint_kwargs = {"cost_fn": cost_fn, "cost_budget": 1.0}
    env = constraint_ctor(env, **constraint_kwargs)
    env = ConstraintMonitor(env)
    env = RewardMonitor(env)
    env = WinningRegionMonitor(env)
    if clip_reward:
        env = ConstraintPersistentGymnasiumWrapper(
            env,
            ClipRewardWrapper
        )
    return env

def make_eval_env(seed=0, max_steps=2000):
    env = Seaquest(
            repeat=4,
            size=(84,84),
            gray=True,
            noops=0,
            lives='reset', 
            sticky=False,
            transpose_obs=True,
            initial_oxygen_depletion_rate=1,
            unexpected_violation=False,
            max_episode_steps=max_steps,
        )
    constraint_ctor = registry.get_constraint("CMDP")
    env = DummyLabelledEnv(env)
    constraint_kwargs = {"cost_fn": cost_fn, "cost_budget": 1.0}
    env = constraint_ctor(env, **constraint_kwargs)
    env = ConstraintMonitor(env)
    env = RewardMonitor(env)
    env = WinningRegionMonitor(env)
    return env


def one_run(run_seed: int, total_timesteps: int, n_eval_episodes: int, log_root: str, tb: bool = False, algo: str = "PPOLag"):
    run_dir = os.path.join(log_root, f"{algo}_seed_{run_seed}")
    os.makedirs(run_dir, exist_ok=True)

    train_env = make_env(seed=run_seed, max_steps=2000)

    algo_cls = {
        "CPO": CPO,
        "PPOLag": PPOLag,
        "TRPOLag": TRPOLag
    }[algo]

    base_kwargs = {
        "tensorboard_logdir": (str(run_dir) if tb else None),
        "seed": run_seed,
        "device": "auto",
        "verbose": 0,
        "env_fn": make_eval_env,
    }

    algo_kwargs = {
        "policy_kwargs": {
            "features_extractor_class": NatureCNN,
            "features_extractor_kwargs": {
                "grayscale_obs": True,
                "normalize_images": True,
            } 
        }
    }

    model = algo_cls(train_env, **base_kwargs, **algo_kwargs)

    run_kwargs = {
        "num_eval_episodes": n_eval_episodes,
        "eval_freq": 100_000,
        "log_freq": 100_000,
        "prefill": 0,
        "save_freq": 0,
        "stats_window_size": 100,
        "stats_window_overrides": {"train/rollout/satisfied": 2000, "eval/rollout/satisfied": 200}
    }

    model.train(total_timesteps, **run_kwargs)

    # load results from tensorboard and return

    return True
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--runs", type=int, default=5)
    parser.add_argument("-t", "--timesteps", type=int, default=10_000_000)
    parser.add_argument("-e", "--eval", type=int, default=20)
    parser.add_argument("--logdir", type=str, default="./logdir/seaquest")
    parser.add_argument("--tensorboard", action="store_true", default=False)
    parser.add_argument("--algo", type=str, choices=["CPO", "PPOLag", "TRPOLag"], default="PPOLag")
    args = parser.parse_args()

    runs = args.runs
    timesteps = args.timesteps
    eval_episodes = args.eval
    algo_id = args.algo

    log_root = os.path.join(args.logdir, f"{algo_id}")
    os.makedirs(log_root, exist_ok=True)

    seeds = [i for i in range(1, runs)]

    for i, seed in enumerate(seeds, 0):
        print(f"\n=== Run {i+1}/{runs} (seed={seed}) ===")
        res = one_run(seed, timesteps, eval_episodes, log_root, tb=args.tensorboard, algo=algo_id)

        assert res, "run failed"