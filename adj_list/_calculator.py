from typing import Union

import numpy as np
import matplotlib.pyplot as plt


class __Calculator:
    def __init__(
        self, adj_list: dict[int, list[int]], node_num: int, node_radius: float
    ):
        self.adj_list = adj_list
        self.node_num = node_num
        self.node_radius = node_radius

    @classmethod
    def from_adj_list(cls, adj_list, node_radius: float):
        return cls(adj_list.inner_adj_list, adj_list.node_num, node_radius)


class NodeCalculator(__Calculator):

    def __init__(
        self, adj_list: dict[int, list[int]], node_num: int, node_radius: float
    ):
        super().__init__(adj_list, node_num, node_radius)

    def cal_wedge_angle(self, node_idx: int) -> tuple[float, float]:
        wedge_center = self.cal_wedge_center(node_idx)
        wedge_length = self.cal_wedge_length()
        start_angle = wedge_center - wedge_length / 2
        end_angle = wedge_center + wedge_length / 2
        return start_angle, end_angle

    def cal_wedge_center(self, node_idx: int) -> float:
        return 2 * np.pi * (node_idx / self.node_num)

    def cal_wedge_length(self):
        return 2 * np.pi * (1 / self.node_num)

    def polar_2_rect(self, node_idx: int) -> tuple[float, float]:
        x = np.cos(self.cal_wedge_center(node_idx)) * self.node_radius
        y = np.sin(self.cal_wedge_center(node_idx)) * self.node_radius
        return x, y


class EdgeCalculator(__Calculator):
    def __init__(
        self, adj_list: dict[int, list[int]], node_num: int, node_radius: float
    ):
        super().__init__(adj_list, node_num, node_radius)
        self.node_calc = NodeCalculator(self.adj_list, node_num, node_radius)

    def cal_edge_center(self, node_idx_a: int, node_idx_b: int) -> tuple[float, float]:
        phi_a = self.node_calc.cal_wedge_center(node_idx_a)
        phi_b = self.node_calc.cal_wedge_center(node_idx_b)
        if (phi_a - phi_b) % np.pi == 0:
            return 0, 0
        cos_a, sin_a = np.cos(phi_a), np.sin(phi_a)
        cos_b, sin_b = np.cos(phi_b), np.sin(phi_b)
        numerator = 2 * (cos_a * cos_b + sin_a * sin_b)
        dominator = (cos_a + cos_b) ** 2 + (sin_a + sin_b) ** 2
        mu = 1 - numerator / dominator
        mu *= self.node_radius
        edge_center_x = mu * (cos_a + cos_b)
        edge_center_y = mu * (sin_b + sin_a)
        return edge_center_x, edge_center_y

    @staticmethod
    def cal_vector_angle(
        vector_start: tuple[float, float], vector_end: tuple[float, float]
    ) -> float:
        vector_start_x, vector_start_y = vector_start
        vector_end_x, vector_end_y = vector_end
        vector_x = vector_end_x - vector_start_x
        vector_y = vector_end_y - vector_start_y
        vector_angle = np.arctan2(vector_y, vector_x)
        return vector_angle

    def cal_edge_radius(self, node_idx_from: int, node_idx_to: int) -> float:
        start_angle = self.node_calc.cal_wedge_center(node_idx_from)
        end_angle = self.node_calc.cal_wedge_center(node_idx_to)
        edge_radius = self.node_radius * np.tan((end_angle - start_angle) / 2)
        return np.abs(edge_radius)

    def cal_edge_angle(
        self, node_idx_from: int, node_idx_to: int
    ) -> Union[tuple[float, float], tuple[tuple[float, float], tuple[float, float]]]:
        edge_center = self.cal_edge_center(node_idx_from, node_idx_to)
        if edge_center != (0, 0):
            start_angle = self.cal_vector_angle(
                edge_center, self.node_calc.polar_2_rect(node_idx_from)
            )
            end_angle = self.cal_vector_angle(
                edge_center, self.node_calc.polar_2_rect(node_idx_to)
            )
        else:
            start_angle = self.node_calc.polar_2_rect(node_idx_from)
            end_angle = self.node_calc.polar_2_rect(node_idx_to)
        return start_angle, end_angle


if __name__ == "__main__":
    from adj_list._adj_list import ADJList

    adj_list_ = ADJList(6)
    node_radius_ = 1
    scale = 2
    node_idx_a_ = 0
    node_idx_b_ = 2

    edge_calc = EdgeCalculator.from_adj_list(adj_list_, node_radius_)

    node_0 = np.array(edge_calc.node_calc.polar_2_rect(node_idx_a_))
    node_1 = np.array(edge_calc.node_calc.polar_2_rect(node_idx_b_))

    edge_center_ = edge_calc.cal_edge_center(node_idx_a_, node_idx_b_)

    theta = np.linspace(0, 2 * np.pi, 100)
    x_ = np.cos(theta) * node_radius_
    y_ = np.sin(theta) * node_radius_

    fig = plt.figure(figsize=(scale * node_radius_, scale * node_radius_))
    plt.scatter(*node_0, c="red")
    plt.scatter(*node_1, c="red")
    plt.scatter(*edge_center_, c="blue")
    plt.plot(x_, y_)
    plt.xlim(-scale * node_radius_, scale * node_radius_)
    plt.ylim(-scale * node_radius_, scale * node_radius_)
    plt.show()
