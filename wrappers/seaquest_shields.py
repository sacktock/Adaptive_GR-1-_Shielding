import numpy as np
import math
import gymnasium as gym
from gymnasium.utils import seeding

class Naive_Shield():

    def __init__(self):
        self.predicates = ['_up', '_down', '_depth', '_oxygen_level']
        self.oxygen_level = None
        self.np_random = None

    def __call__(self, action):
        if self.oxygen_level <= 25 + 4:
            if action in [2,6,7,10,14,15]:
                return action, False
            else:
                action = self.np_random.choice([2,6,7,10,14,15])
                return action, True

        return action, False

    def seed(self, seed=None):
        if seed is None:
            pass
        self.np_random, _ = seeding.np_random(seed)

    def reset(self, seed=seed):
        self.seed(seed)

    def step(self, info):
        abstr = {k: info[k] for k in self.predicates}
        self.oxygen_level = abstr['_oxygen_level']

    def repair(self):
        raise NotImplementedError

class Static_Shield():

    def __init__(self):
        self.predicates = ['_up', '_down', '_depth', '_oxygen_level']
        self.oxygen_level = None
        self.depth = None
        self.np_random = None

    def __call__(self, action):
        time_until_surface = math.ceil(self.depth/4)

        if self.oxygen_level <= time_until_surface + 5:
            if action in [2,6,7,10,14,15]:
                return action, False
            else:
                action = self.np_random.choice([2,6,7,10,14,15])
                return action, True
        
        return action, False

    def seed(self, seed=None):
        if seed is None:
            pass
        self.np_random, _ = seeding.np_random(seed)

    def reset(self, seed=seed):
        self.seed(seed)

    def step(self, info):
        abstr = {k: info[k] for k in self.predicates}
        self.oxygen_level = abstr['_oxygen_level']
        self.depth = abstr['_depth']

    def repair(self):
        raise NotImplementedError

class Repaired_Shield():

    def __init__(self):
        self.predicates = ['_up', '_down', '_depth', '_oxygen_level']
        self.oxygen_level = None
        self.depth = None
        self.np_random = None

    def __call__(self, action):
        time_until_surface = math.ceil(self.depth/4)

        if self.oxygen_level//2 <= time_until_surface + 5:
            if action in [2,6,7,10,14,15]:
                return action, False
            else:
                action = self.np_random.choice([2,6,7,10,14,15])
                return action, True
        
        return action, False

    def seed(self, seed=None):
        if seed is None:
            pass
        self.np_random, _ = seeding.np_random(seed)

    def reset(self, seed=seed):
        self.seed(seed)

    def step(self, info):
        abstr = {k: info[k] for k in self.predicates}
        self.oxygen_level = abstr['_oxygen_level']
        self.depth = abstr['_depth']

    def repair(self):
        raise NotImplementedError

class Adaptive_Shield():

    def __init__(self):
        self.predicates = ['_up', '_down', '_depth', '_oxygen_level']
        self.oxygen_level = None
        self._initial_shield = Static_Shield()
        self._repaired_shield = Repaired_Shield()
        self._flag = False

    def __call__(self, action):
        if self._flag:
            return self._repaired_shield(action)
        else:
            return self._initial_shield(action)

    def seed(self, seed=None):
        self._initial_shield.seed(seed=seed)
        self._repaired_shield.seed(seed=seed)

    def reset(self, seed=None):
        self._initial_shield.reset(seed=seed)
        self._repaired_shield.reset(seed=seed)
        self._flag = False

    def step(self, info):
        abstr = {k: info[k] for k in self.predicates}
        oxygen_level = abstr['_oxygen_level']
        if (self.oxygen_level is not None) and (self.oxygen_level - oxygen_level >= 2):
            self._flag = True

        self.oxygen_level = abstr['_oxygen_level']

        self._initial_shield.step(info)
        self._repaired_shield.step(info)
        
    def repair(self):
        raise NotImplementedError



        
