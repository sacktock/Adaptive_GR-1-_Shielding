
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

class MinePumpEnvV1(gym.Env):
    """
    Minepump with methane, Markovian inflow, and 2-level tariff.

    Observations (spaces.Dict):
        - methane: Discrete(2)   (0=no methane, 1=methane)
        - highwater: Discrete(2) (0=water<H, 1=water>=H)
        - water_level: Box([0], [W_cap])  (float)
        - inflow_state: Discrete(3)  (0=no, 1=low, 2=high)  -- optional for agent
        - price_state: Discrete(2)   (0=low=1, 1=high=4)    -- optional for agent

    Actions:
        Discrete(2): 0 = pump OFF, 1 = pump ON

    Reward (to maximize):
        r_t = - price * action - switch_cost * 1[action != prev_action] - highwater_penalty * 1[highwater]

    Episode termination:
        - No hard termination (water is capped at W_cap).
        - Use time-limit wrapper (recommended), or set max_episode_steps in constructor (truncation).

    Notes:
        - Methane is memoryless: P(ON->OFF)=p_on_to_off, P(OFF->ON)=p_off_to_on (geometric dwell times).
        - Inflow is a 3-state Markov chain over rates {0,0.5,2}.
        - Tariff is a 2-state Markov chain over {1,4} with persistence 0.7 by default.
        - Pump removes R units per step.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        *,
        W_cap: float = 18.0,
        H: float = 10.0,
        R: float = 6.0, # pump removal per step
        # Methane switching probabilities (geometric)
        p_on_to_off: float = 0.25,
        p_off_to_on: float = 1/6,
        # Inflow Markov kernel over states {no, low, high}
        inflow_rates = (0.0, 0.5, 2.0),
        inflow_P: np.ndarray | None = None,
        # Tariff {1,4} with persistence 0.7
        price_low: float = 1.0,
        price_high: float = 4.0,
        p_high_to_high: float = 0.7,
        p_low_to_low: float = 0.7,
        # Reward weights
        switch_on_cost: float = 0.5,
        switch_off_cost: float = 0.1,
        highwater_penalty: float = 1.0,
        # Unexpected violation or not (can highwater and methane be simultaneously true)
        unexpected_violation: bool = False,
        # Control episode length here if desired (otherwise wrap with TimeLimit)
        max_episode_steps: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.W_cap = float(W_cap)
        self.H = float(H)
        self.R = float(R)

        # Sanity checks
        assert 0 < self.H <= self.W_cap
        assert 0.0 <= p_on_to_off <= 1.0 and 0.0 <= p_off_to_on <= 1.0
        self.p_on_to_off = float(p_on_to_off)
        self.p_off_to_on = float(p_off_to_on)

        self.inflow_rates = np.array(inflow_rates, dtype=np.float32)
        assert self.inflow_rates.shape == (3,), "inflow_rates must be length-3."

        if inflow_P is None:
            # Default kernel
            inflow_P = np.array([
                [0.80, 0.18, 0.02],
                [0.10, 0.80, 0.10],
                [0.02, 0.28, 0.70]
            ], dtype=np.float32)
        else:
            inflow_P = np.array(inflow_P, dtype=np.float32)
        assert inflow_P.shape == (3, 3), "inflow_P must be 3x3."
        assert np.all(inflow_P.sum(axis=1) == 1), "inflow_P is not a stochastic matrix!"

        self.inflow_P = inflow_P

        self.price_low = float(price_low)
        self.price_high = float(price_high)
        self.p_high_to_high = float(p_high_to_high)
        self.p_low_to_low = float(p_low_to_low)

        self.switch_on_cost = float(switch_on_cost)
        self.switch_off_cost = float(switch_off_cost)
        self.highwater_penalty = float(highwater_penalty)

        self.unexpected_violation = unexpected_violation

        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        # Act/Obs space
        self.action_space = spaces.Discrete(2) 
        self.observation_space = spaces.Dict({
            "methane": spaces.Discrete(2),
            "highwater": spaces.Discrete(2),
            "prev_methane": spaces.Discrete(2),
            "prev_highwater": spaces.Discrete(2),
            "water_level": spaces.Box(low=np.array([0.0], dtype=np.float32),
                                      high=np.array([self.W_cap], dtype=np.float32),
                                      dtype=np.float32),
            "inflow_state": spaces.Discrete(3),
            "price_state": spaces.Discrete(2),
        })

        # Internals
        self.np_random = None
        self._t = 0
        self._obs_buffer = {
            "methane": 0,
            "highwater": 0,
        }
        self._water = 0.0
        self._methane = 0
        self._inflow_state = 0 
        self._price_state = 0
        self._prev_action = 0
        self._guarantee_1 = True
        self._guarantee_2 = True
        self._ep_safe = True

    def seed(self, seed: int | None = None):
        self.reset(seed=seed)

    def _price_value(self, state: int) -> float:
        return self.price_high if state == 1 else self.price_low

    def _obs(self):
        highwater = 1 if self._water > self.H else 0
        return {
            "methane": int(self._methane),
            "highwater": int(highwater),
            "prev_methane": int(self._obs_buffer["methane"]),
            "prev_highwater": int(self._obs_buffer["highwater"]),
            "water_level": np.array([self._water], dtype=np.float32),
            "inflow_state": int(self._inflow_state),
            "price_state": int(self._price_state),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.np_random, _ = seeding.np_random(seed)

        self._t = 0
        self._obs_buffer = { # reset the obs buffer
            "methane": 0,
            "highwater": 0,
        }
        self._water = float(self.np_random.uniform(0.0, self.H))  # start below highwater
        self._methane = 0                                         # start without methane
        self._inflow_state = int(self.np_random.integers(0, 3))   # random of {0,1,2}
        self._price_state = int(self.np_random.integers(0, 2))    # random {low, high}
        self._prev_action = 0
        self._guarantee_1 = True
        self._guarantee_2 = True
        self._ep_safe = True

        info = {
            "highwater": "false",
            "methane": "false",
            "pump": "false",
            "guarantee_1": self._guarantee_1,
            "guarantee_2": self._guarantee_2,
            "in_winning_region": self._guarantee_1 and self._guarantee_2,
            "is_success_1": self._guarantee_1,
            "is_success_2": self._guarantee_2,
            "is_success": self._guarantee_1 and self._guarantee_2,
        }

        return self._obs(), info

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action. {action}"

        guarantee_1 = not (((self._obs_buffer["highwater"]==1) and (self._obs_buffer["methane"] == 0)) and (action==0))
        self._guarantee_1 = bool(guarantee_1 * self._guarantee_1)

        guarantee_2 = not ((self._obs_buffer["methane"]==1) and (action==1))
        self._guarantee_2 = bool(guarantee_2 * self._guarantee_2)

        # save past methane and highwater state
        self._obs_buffer = {
            "methane": self._methane,
            "highwater": 1 if self._water > self.H else 0
        }

        # update tariff
        if self._price_state == 1:
            stay_high = self.np_random.random() < self.p_high_to_high
            self._price_state = 1 if stay_high else 0
        else:
            stay_low = self.np_random.random() < self.p_low_to_low
            self._price_state = 0 if stay_low else 1
        price = self._price_value(self._price_state)

        # update methane ON/OFF
        if self._methane == 1:
            if self.np_random.random() < self.p_on_to_off:
                self._methane = 0
        else:
            if self.np_random.random() < self.p_off_to_on:
                self._methane = 1

        # update inflow
        probs = self.inflow_P[self._inflow_state]
        self._inflow_state = int(self.np_random.choice(3, p=probs))
        inflow = self.inflow_rates[self._inflow_state]

        # update water level
        pump = int(action)
        delta = inflow - (self.R if pump == 1 else 0.0)
        self._water = float(np.clip(self._water + delta, 0.0, self.W_cap))

        # compute reward
        if pump != self._prev_action:
            if pump == 0:
                switch_pen = self.switch_on_cost 
            else:
                switch_pen = self.switch_off_cost
        else:
            switch_pen = 0.0
        highwater = 1 if self._water > self.H else 0
        r = - price * pump - switch_pen - self.highwater_penalty * highwater

        if highwater and not self.unexpected_violation:
            self._methane = 0 

        # evolve internals
        self._prev_action = pump
        self._t += 1

        # No terminal condition (use TimeLimit), but allow optional truncation:
        terminated = False
        truncated = False
        if self.max_episode_steps is not None and self._t >= self.max_episode_steps:
            truncated = True

        info = {
            "highwater": "true" if highwater else "false",
            "methane": "true" if self._methane == 1 else "false",
            "pump": "true" if pump == 1 else "false",
            "guarantee_1": guarantee_1,
            "guarantee_2": guarantee_2,
            "in_winning_region": guarantee_1 and guarantee_2,
            "is_success_1": self._guarantee_1,
            "is_success_2": self._guarantee_2,
            "is_success": self._guarantee_1 and self._guarantee_2,
        }

        self.render()

        return self._obs(), float(r), terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        print(
            f"t={self._t:4d} | water={self._water:5.2f} (H={self.H}) "
            f"| methane={self._methane} | inflow_state={self._inflow_state} "
            f"| price_state={self._price_state} (C={self._price_value(self._price_state)})"
        )

    def close(self):
        pass
