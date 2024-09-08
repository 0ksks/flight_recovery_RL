from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import toml

from ._calculator import EdgeCalculator, NodeCalculator
from ._drawer import (
    NodeDrawer,
    EdgeDrawer,
    SupportNodeType,
    SupportEdgeType,
    NodeFont,
    EdgeFont,
)

from typing import Optional


class ADJList:
    def __init__(self, node_num: int, density: float = 0.5, seed=1):
        self.node_num = node_num
        self.seed = seed
        self.inner_adj_list = self.__gen_adj_list(density)
        self.__face_width = None
        self.__edge_calc: Optional[EdgeCalculator] = None
        self.__edge_drawer: Optional[EdgeDrawer] = None
        self.__node_calc: Optional[NodeCalculator] = None
        self.__node_drawer: Optional[NodeDrawer] = None
        self.node_element_array = None
        self.edge_element_matrix = None

    def __register_calculator(self, node_radius: float):
        self.__edge_calc = EdgeCalculator.from_adj_list(
            self, node_radius - self.__face_width
        )
        self.__node_calc = NodeCalculator.from_adj_list(self, node_radius)

    def __gen_adj_list(self, density: float):
        np.random.seed(self.seed)
        inner_adj_list = {}
        for node_from in range(self.node_num):
            inner_adj_list[node_from] = []
            for node_to in range(node_from + 1, self.node_num):
                if np.random.random() < density:
                    inner_adj_list[node_from].append(node_to)
        return inner_adj_list

    def register_drawer(self, fig_size: int, node_radius: float, face_width: float):
        self.__face_width = face_width
        self.__register_calculator(node_radius)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        ax.set_aspect("equal", "box")
        ax.set_xlim([-node_radius, node_radius])
        ax.set_ylim([-node_radius, node_radius])
        self.__node_drawer = NodeDrawer(node_radius)
        self.__edge_drawer = EdgeDrawer()
        return fig, ax

    def config_node_font(self, node_type: SupportNodeType, font: NodeFont) -> None:
        self.__node_drawer.font[node_type] = font

    def get_node_element(self, node_type: SupportNodeType, node_idx: int):
        start_angle, end_angle = self.__node_calc.cal_wedge_angle(node_idx)
        return self.__node_drawer.draw(
            node_type, start_angle, end_angle, self.__face_width
        )

    def config_edge_font(self, edge_type: SupportEdgeType, font: EdgeFont) -> None:
        self.__edge_drawer.font[edge_type] = font

    def config_font_from_toml(self, toml_file_path: str) -> None:
        with open(toml_file_path, "r") as f:
            config = toml.load(f)

        def get_all(config: dict[str, dict]):
            def get_by_type(config: dict[str, dict], type_: str):
                keys = type_.split(".")
                result = config[keys[0]][keys[1]]
                for k, v in config[keys[0]]["common"].items():
                    result[k] = v
                return result

            node_types = config["node"].keys()
            edge_types = config["edge"].keys()

            node_result = {}
            edge_result = {}

            for node_type in node_types:
                if node_type != "common":
                    node_result[node_type] = get_by_type(config, f"node.{node_type}")

            for edge_type in edge_types:
                if edge_type != "common":
                    edge_result[edge_type] = get_by_type(config, f"edge.{edge_type}")

            return node_result, edge_result

        node_config, edge_config = get_all(config)
        for k, v in node_config.items():
            self.config_node_font(k, NodeFont(**v))
        for k, v in edge_config.items():
            self.config_edge_font(k, EdgeFont(**v))

    def get_edge_element(
        self, edge_type: SupportEdgeType, node_idx_from: int, node_idx_to: int
    ):
        edge_center = self.__edge_calc.cal_edge_center(node_idx_from, node_idx_to)
        edge_start_angle, edge_end_angle = self.__edge_calc.cal_edge_angle(
            node_idx_from, node_idx_to
        )
        if edge_center != (0, 0):
            edge_radius = self.__edge_calc.cal_edge_radius(node_idx_from, node_idx_to)
        else:
            edge_radius = 0
        return self.__edge_drawer.draw(
            edge_type, edge_center, edge_radius, edge_start_angle, edge_end_angle
        )

    @staticmethod
    def add_element(ax, element, add_arrow=False):
        if isinstance(element, tuple):  # edge
            line, arrow = element
            ax.add_artist(line)
            if add_arrow:
                ax.add_artist(arrow)
        else:  # node
            ax.add_artist(element)

    @staticmethod
    def add_elements(ax, elements, add_arrow=False):
        for element in elements:
            ADJList.add_element(ax, element, add_arrow)

    def ini_elements(self, ini_edge=True):
        node_element_array = []
        if ini_edge:
            edge_element_matrix = defaultdict(list)
        for node_idx in range(self.node_num):
            node_element_array.append(self.get_node_element("unvisited", node_idx))
            node_idx_from = node_idx
            if ini_edge:
                for node_idx_to in self.inner_adj_list[node_idx]:
                    edge_element_matrix[node_idx_from].append(
                        self.get_edge_element("unvisited", node_idx_from, node_idx_to)
                    )
        self.node_element_array = node_element_array
        if ini_edge:
            self.edge_element_matrix = edge_element_matrix

    def ini_fig(self, ax, add_arrow=False, ini_edge=True):
        self.add_elements(ax, self.node_element_array, add_arrow)
        if ini_edge:
            for edge_element_array in self.edge_element_matrix.values():
                self.add_elements(ax, edge_element_array, add_arrow)

    def get_trajectory_elements(self, trajectory: list[int]):
        traj_elements = [self.get_node_element("visited", trajectory[0])]
        for traj_idx in range(1, len(trajectory)):
            traj_elements.append(
                self.get_edge_element(
                    "visited", trajectory[traj_idx - 1], trajectory[traj_idx]
                )
            )
            traj_elements.append(self.get_node_element("visited", trajectory[traj_idx]))
        return traj_elements

    def add_trajectory_elements(self, ax, trajectory: list[int], add_arrow=False):
        traj_elements = self.get_trajectory_elements(trajectory)
        self.add_elements(ax, traj_elements, add_arrow)
