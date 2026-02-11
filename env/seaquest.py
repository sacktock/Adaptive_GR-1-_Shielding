
from __future__ import annotations
import math
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TimeLimit
import ale_py
from typing import SupportsFloat

gym.register_envs(ale_py)

class OxygenWrapper(gym.Wrapper):

    def __init__(self, env, initial_oxygen_depletion_rate=1, unexpected_violation=True):
        super().__init__(env)
        self.oxygen_level = None
        self.depth = None
        self.oxygen_depletion_rate = int(initial_oxygen_depletion_rate)
        self.unexpected_violation = unexpected_violation
        if self.unexpected_violation:
            self.lower_depletion_rate = int(initial_oxygen_depletion_rate)
            self.higher_depletion_rate = 2
            self.switching_oxygen_level = 48
            self._switching_flag = True
        else:
            self._switching_flag = False
        self.steps_below_surface = None
        self.steps_on_the_surface = None
        self.faster_rate = False

        self._last_lives = None
        self._flag_1 = False

    def toggle_flag(self):
        self._flag_1 = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ram = self.env.unwrapped.ale.getRAM()

        prev_depth = self.depth
        self.depth = int(ram[97]) - 13

        # reset the flashing flag after life lost
        current = self.env.unwrapped.ale.lives()
        if current < self._last_lives and self.depth == 0:
            #assert self._flag_1, f"_flag_1 {self._flag_1}, current {current}, _last_lives {self._last_lives}"
            self._flag_1 = False
            self._last_lives = current

        if (self.depth > 0) or (prev_depth > 0):# and (not self._flag_1) and (not terminated) and (not truncated):
            self.steps_below_surface += 1
            self.steps_on_the_surface = 0
            if self.steps_below_surface > 0 and (self.steps_below_surface % (4 // self.oxygen_depletion_rate)) == 0: 
                self.oxygen_level = max(self.oxygen_level - 1, 0)
    
        if self.depth == 0:
            self._flag_1 = False
            self.steps_on_the_surface += 1
            self.steps_below_surface = 0
            if self.steps_on_the_surface > 0 and (self.steps_on_the_surface % 4) == 0 and not self.reset_flag:
                self.reset_flag = False
                self.oxygen_level = min(self.oxygen_level + 1, 64) 
            if self.steps_on_the_surface > 0 and (self.steps_on_the_surface % 4) == 0:
                self.reset_flag = False

        # check if flashing -> if not then don't deplete oxygen
        if np.any(obs == 708):
            self.toggle_flag()
                
        obs[170:175, 48:(48+self.oxygen_level), :] = np.array([214, 214, 214], np.uint8)
        obs[170:175, 112-(64-self.oxygen_level):112, :] = np.array([163, 57, 21], np.uint8)

        if not ((self.oxygen_level > 0) or (self.depth == 0)) or self._flag_1:
            terminated = True

        if (self._switching_flag) and (self.oxygen_level <= self.switching_oxygen_level) and (self.depth > 0) and (self.oxygen_depletion_rate != self.higher_depletion_rate):
            self.oxygen_depletion_rate = self.higher_depletion_rate
            self.faster_rate = True

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        ram = self.env.unwrapped.ale.getRAM()

        self.oxygen_level = 0
        self.depth = 0
        self.steps_below_surface = 0
        self.steps_on_the_surface = 0

        if self.unexpected_violation:
            self.oxygen_depletion_rate = self.lower_depletion_rate
            self.faster_rate = False

        self._last_lives = self.env.unwrapped.ale.lives()
        self._flag_1 = False

        obs[170:175, 48:48+self.oxygen_level, :] = np.array([214, 214, 214], np.uint8)
        obs[170:175, 112-64+self.oxygen_level:112, :] = np.array([163, 57, 21], np.uint8)

        self.reset_flag = True
        
        return obs

class ClipRewardWrapper(gym.RewardWrapper):
    """
    Clip the reward to {+1, 0, -1} by its sign.

    :param env: Environment to wrap
    """

    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        """
        Bin reward to {+1, 0, -1} by its sign.

        :param reward:
        :return:
        """
        return np.sign(float(reward))

class Seaquest(gym.Env):

    def __init__(self, 
        *,
        repeat: int = 4, 
        size: tuple = (84, 84), 
        gray: bool = True, 
        noops: int = 30,
        lives: str = 'unused', 
        sticky: bool = False,
        actions: str = 'all', 
        resize: str = 'opencv', 
        initial_oxygen_depletion_rate: int = 1,
        unexpected_violation: bool = False,
        # Control episode length here if desired (otherwise wrap with TimeLimit)
        max_episode_steps: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()

        # Sanity checks
        assert size[0] == size[1]
        assert lives in ('unused', 'discount', 'reset'), lives
        assert actions in ('all', 'needed'), actions
        assert resize in ('opencv', 'pillow'), resize

        self._env = gym.make("ALE/Seaquest-v5", 
            obs_type="rgb",
            frameskip=1, 
            repeat_action_probability=0.25 if sticky else 0.0, 
            full_action_space=(actions == 'all')
        )

        self._resize = resize
        if self._resize == 'opencv':
            import cv2
            self._cv2 = cv2
        if self._resize == 'pillow':
            from PIL import Image
            self._image = Image

        self.max_episode_steps = max_episode_steps

        self._repeat = repeat
        self._size = size
        self._gray = gray
        self._noops = noops
        self._lives = lives

        self._initial_oxygen_depletion_rate = initial_oxygen_depletion_rate
        self._unexpected_violation = unexpected_violation

        self._env = OxygenWrapper(self._env, initial_oxygen_depletion_rate=self._initial_oxygen_depletion_rate, unexpected_violation=self._unexpected_violation)

        # Act/Obs space
        self.action_space = self._env.action_space
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=size + (1 if self._gray else 3,), dtype=np.uint8)

        self.operational = False
        self.diver_at_depth1 = False
        self.diver_at_depth2 = False
        self.diver_at_depth3 = False
        self.diver_at_depth4 = False

        assert self._env.unwrapped.get_action_meanings()[0] == 'NOOP'
        assert self._env.unwrapped.get_action_meanings()[1] == 'FIRE'

        # Internals
        self.np_random = None
        self._t = 0
        self._buffer = [np.zeros(self._env.observation_space.shape, np.uint8) for _ in range(2)]
        self._ale = self._env.unwrapped.ale
        self._last_lives = None
        self._action = None
        self._ep_safe = True
        self._act_dict = {
            5: 0, 8: 3, 9: 4, 13: 1, 16: 11, 17: 12, 2: 0, 6: 3, 7: 4, 10: 1, 14: 11,15: 12,
        }

    def seed(self, seed: int | None = None):
        self.reset(seed=seed)

    def step(self, action):

        total = 0.0
        dead = False
        over = False

        self._action = action

        info = self._info()
        up = info['_up']
        down = info['_down']
        operational = info['_operational']
        curr_depth = int(info['_depth'])
        prev_depth = int(info['_depth'])
        oxygen_level = int(info['_oxygen_level'])
        t = self._t

        if not operational and (up or down):
            self._action = self._act_dict[self._action]

        if curr_depth == 92 and down:
            self._action = self._act_dict[self._action]

        if curr_depth == 20 and self.diver_at_depth1:
            self.diver_at_depth1 = False
            self.diver_at_depth2 = True
            self.diver_at_depth3 = True
            self.diver_at_depth4 = True

        if curr_depth == 44 and self.diver_at_depth2:
            self.diver_at_depth1 = True
            self.diver_at_depth2 = False
            self.diver_at_depth3 = True
            self.diver_at_depth4 = True

        if curr_depth == 68 and self.diver_at_depth3:
            self.diver_at_depth1 = True
            self.diver_at_depth2 = True
            self.diver_at_depth3 = False
            self.diver_at_depth4 = True

        if curr_depth == 92 and self.diver_at_depth4:
            self.diver_at_depth1 = True
            self.diver_at_depth2 = True
            self.diver_at_depth3 = True
            self.diver_at_depth4 = False

        for repeat in range(self._repeat):
            ob, reward, over, _, _ = self._env.step(self._action)

            self._t += 1
            total += reward
            if repeat == self._repeat - 2:
                self._buffer[1] = np.array(ob, dtype=np.uint8)
            if over:
                break
            if self._lives != 'unused':
                current = self._ale.lives()
                if current < self._last_lives:
                    dead = True
                    self._last_lives = current
                    break

        info = self._info()
        depth = int(info['_depth'])
        oxygen_level = int(info['_oxygen_level'])

        if depth == 0:
            self.operational = False

        if oxygen_level == 64:
            self.operational = True
            self.diver_at_depth1 = True
            self.diver_at_depth2 = True
            self.diver_at_depth3 = True
            self.diver_at_depth4 = True

        # we might need an operational flag 
        # prevent agent from going down if first down action does not result in depth decrease from depth == 0
        # assert not up or ((prev_depth - depth) == 4) or depth==0 or over or not operational, f"up={up}, curr_depth={prev_depth}, depth={depth}, over={over}, dead={dead}, t={t}, self._t={self._t}, action={self._action}, operational={operational}"
        # assert not down or ((depth - prev_depth) == 4) or depth==92 or over or not operational, f"down={down}, curr_depth={prev_depth}, depth={depth}, over={over}, dead={dead}, t={t}, self._t={self._t}, action={self._action}, operational={operational}"

        if not self._repeat:
            self._buffer[1][:] = self._buffer[0][:]

        self._buffer[0] = np.array(ob, dtype=np.uint8)

        terminated = dead or over
        truncated = (self.max_episode_steps and self._t >= self.max_episode_steps)

        # auto reset on collision with shark or submarine
        terminated = terminated or not (not up or ((prev_depth - depth) == 4) or depth==0 or over or not operational)\
            or not (not down or ((depth - prev_depth) == 4) or depth==92 or over or not operational)

        info = self._info()

        return self._obs(), total, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        self._t = 0
        self._env.reset(seed=seed, options=options)
        self._action = 1
        ob, _, dead, _, _ = self._env.step(self._action)
        self._action = 0
        if self._noops:
            for _ in range(self._noops):
                ob, _, dead, _, _ = self._env.step(self._action)
                if dead:
                    self._env.reset(seed=seed+1, options=options)
        
        self._last_lives = self._ale.lives()
        self._buffer[0] = np.array(ob, dtype=np.uint8)
        self._buffer[1].fill(0)
        self._ep_safe = True
        self.operational = False
        self.diver_at_depth1 = False
        self.diver_at_depth2 = False
        self.diver_at_depth3 = False
        self.diver_at_depth4 = False
        info = self._info()
        return self._obs(), info

    def _obs(self):
        np.maximum(self._buffer[0], self._buffer[1], out=self._buffer[0])
        image = self._buffer[0]
        if image.shape[:2] != self._size:
            if self._resize == 'opencv':
                image = self._cv2.resize(
                    image, self._size, interpolation=self._cv2.INTER_AREA)
        if self._resize == 'pillow':
            image = self._image.fromarray(image)
            image = image.resize(self._size, self._image.NEAREST)
            image = np.array(image)
        if self._gray:
            weights = [0.299, 0.587, 1 - (0.299 + 0.587)]
            image = np.tensordot(image, weights, (-1, 0)).astype(image.dtype)
            image = image[:, :, None]

        return image

    def _info(self):

        up = self._action in [2,6,7,10,14,15]
        down = self._action in [5,8,9,13,16,17]
        depth = self._env.depth
        oxygen_level = self._env.oxygen_level
        operational = self.operational
        faster_rate = self._env.faster_rate
        reset_flag = self._env.reset_flag

        safe = (oxygen_level > 0) or (depth == 0)

        assert safe, f"depth={depth}, oxygen_level={oxygen_level}"

        self._ep_safe = bool(safe * self._ep_safe)

        if not self._unexpected_violation or \
            (self._unexpected_violation and \
                (self._env.oxygen_depletion_rate == self._initial_oxygen_depletion_rate)):
            time_until_surface = math.ceil(depth/4)
            if oxygen_level < time_until_surface + 1:
                in_winning_region = False
            elif oxygen_level <= time_until_surface + 5:
                in_winning_region = up
            else:
                in_winning_region = True
        else:
            time_until_surface = math.ceil(depth/4)
            if oxygen_level//2 < time_until_surface + 1:
                in_winning_region = False
            elif oxygen_level//2 <= time_until_surface + 5:
                in_winning_region = up
            else:
                in_winning_region = True

        oxygen_bits = [(bool((oxygen_level >> i) & 1)) for i in range(7)]
        oxygen_encoding = {f"oxygen{i}": "true" if oxygen_bits[i] else "false" for i in range(7)}

        depth_idx = depth // 4
        depth_bits = [(bool((depth_idx >> i) & 1)) for i in range(5)]
        depth_encoding = {f"depth{i}": "true" if depth_bits[i] else "false" for i in range(5)}

        diver_encodings = {f"diver_at_depth{i}": "true" if getattr(self, f"diver_at_depth{i}", None) else "false" for i in range(1, 5)}

        info = {
            '_up': up,
            '_down': down, 
            "up": "true" if up else "false",
            "down": "true" if down else "false",
            '_depth': depth, 
            '_oxygen_level': oxygen_level, 
            "operational": "true" if operational else "false",
            "reset_flag": "true" if reset_flag else "false",
            "faster_rate": "true" if faster_rate else "false",
            '_operational': operational,
            'guarantee_1': safe,
            'in_winning_region': in_winning_region,
            'is_success_1': self._ep_safe,
            'is_success': self._ep_safe,
            **oxygen_encoding,
            **depth_encoding,
            **diver_encodings}
        return info


    def close(self):
        return self._env.close()

        