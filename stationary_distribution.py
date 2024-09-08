import numpy as np
from adj_list import ADJList


class StationaryDistribution:
    def __init__(self, adj_list: dict):
        self.adj_list = adj_list
        self.num_states = len(self.adj_list)
        self.__stationary_distribution = None

    @classmethod
    def from_adj_list(cls, adj_list: ADJList):
        return cls(adj_list.inner_adj_list)

    def __getitem__(self, item):
        return self.__stationary_distribution[item]

    def __repr__(self):
        return self.__stationary_distribution.__repr__()

    def transition_matrix(self, from_state: int, to_state: int, epsilon=1e-9) -> float:
        if to_state not in self.adj_list[from_state]:
            return epsilon
        return 1 / len(self.adj_list[from_state])

    def __iter_calculation(self, max_iterations, tolerance):
        pi = np.ones(self.num_states) / self.num_states

        for iteration in range(max_iterations):
            pi_new = np.zeros(self.num_states)

            for j in range(self.num_states):
                for i in range(self.num_states):
                    pi_new[j] += pi[i] * self.transition_matrix(i, j)

            if np.linalg.norm(pi_new - pi) < tolerance:
                break

            pi = pi_new

        return pi

    def calculate(self, max_iterations=10000, tolerance=1e-9):
        self.__stationary_distribution = self.__iter_calculation(
            max_iterations, tolerance
        )
