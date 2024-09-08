import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch as th
from typing import Optional


class DagEnv(gym.Env):
    def __init__(self, adj_list: dict[int, list[int]]):
        super(DagEnv, self).__init__()
        self.new_visit_reward = 1.0
        self.adj_list = adj_list
        self.n_nodes = len(adj_list)

        # Observation space: Discrete space for each node (0 or 1) and current state
        self.observation_space = spaces.Dict({
            "visited_state"  : spaces.MultiBinary(self.n_nodes),
            "current_state": spaces.Discrete(self.n_nodes)
        })

        # Action space: One-hot vector representing the node to visit
        self.action_space = spaces.Discrete(self.n_nodes)

        # Reward space: Scalar reward bounded by [0, +inf)
        self.reward_range = (0, float("inf"))

        # 初始化 observation 属性
        self.observation = None

    @classmethod
    def from_adj_list(cls, node_num: int, density: float = 0.5, seed=1):
        from adj_list import ADJList

        adj_list = ADJList(node_num, density, seed)
        return cls(adj_list.inner_adj_list)

    @staticmethod
    def get_action_idx(action: int) -> int:
        return action

    @staticmethod
    def seed(seed: Optional[int] = None):
        np.random.seed(seed)
        th.manual_seed(seed)

    def reset(self, **kwargs):
        visited_state = np.zeros(self.n_nodes, dtype=np.int64)
        current_state = 0
        self.observation = {"visited_state": visited_state, "current_state": current_state}  # 初始化 observation
        return self.observation, {}

    def step(self, action: int):
        visited_state = self.observation["visited_state"].copy()
        current_state = self.observation["current_state"]

        if self._is_visited(visited_state, action) or not self._is_reachable(current_state, action):
            done = True
            reward = 0.0
        else:
            reward = self.new_visit_reward
            done = False
            visited_state[action] = 1
            current_state = action

        self.observation = {"visited_state": visited_state, "current_state": current_state}
        return self.observation, reward, done, {}

    @staticmethod
    def _is_visited(visited_state, action) -> bool:
        return visited_state[action] == 1

    def _is_reachable(self, current_state, action) -> bool:
        neighbours = self.adj_list[current_state]
        return action in neighbours

    def get_action_mask(self):
        return th.tensor(self.observation["visited_state"]==0).to(dtype=th.bool)
