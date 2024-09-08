import matplotlib.pyplot as plt
import matplotlib.patches as pat
import numpy as np
from dataclasses import dataclass, asdict
from typing import Union, Literal, Any

SupportNodeType = Literal["visited", "unvisited", "source", "sink", "unreachable"]
SupportEdgeType = Literal["visited", "unvisited", "reachable", "unreachable"]
Line = plt.Line2D
Patch = Union[pat.Wedge, pat.Arc, pat.FancyArrowPatch]


@dataclass
class NodeFont:
    """
    - **face_color**: color
    - **border_width**: float
    - **border_color**: color
    """

    face_color: Any
    border_width: float
    border_color: Any

    def __getitem__(self, key):
        return getattr(self, key)


@dataclass
class EdgeFont:
    """
    - **edge_width**: float
    - **edge_color**: color
    - **arrow_width**: float
    - **arrow_length**: float
    - **arrow_color**: color
    - **arrow_pos**: float
    """

    edge_width: float
    edge_color: Any
    arrow_width: float
    arrow_length: float
    arrow_color: Any
    arrow_pos: float

    def __getitem__(self, key):
        return getattr(self, key)


class Drawer:

    @staticmethod
    def draw_wedge(
        radius,
        angle_start,
        angle_end,
        face_width,
        face_color,
        border_width,
        border_color,
    ) -> Patch:
        wedge = pat.Wedge(
            center=(0, 0),
            r=radius,
            theta1=np.degrees(angle_start),
            theta2=np.degrees(angle_end),
            width=face_width,
            facecolor=face_color,
            linewidth=border_width,
            edgecolor=border_color,
        )
        return wedge

    @staticmethod
    def draw_arrow_arc(
        center,
        radius,
        angle_start,
        angle_end,
        edge_width,
        edge_color,
        arrow_width,
        arrow_length,
        arrow_color,
        arrow_pos: float = 1.0,
    ) -> Union[tuple[Patch, Patch], tuple[Line, Patch]]:
        if center != (0, 0):
            reverse = False
            if angle_start > angle_end:
                angle_start, angle_end = angle_end, angle_start
                reverse = not reverse
            if angle_end - angle_start > np.pi:
                angle_start, angle_end = angle_end, angle_start + 2 * np.pi
                reverse = not reverse
            if reverse:
                arrow_pos = 1 - arrow_pos
            arrow_angle = angle_start * (1 - arrow_pos) + angle_end * arrow_pos
            arrow_x = center[0] + radius * np.cos(arrow_angle)
            arrow_y = center[1] + radius * np.sin(arrow_angle)

            dx = -np.sin(arrow_angle)
            dy = np.cos(arrow_angle)

            if reverse:
                dx, dy = -dx, -dy

            if angle_start > angle_end:
                arrow_start = (arrow_x + arrow_length * dx, arrow_y + arrow_length * dy)
                arrow_end = (arrow_x - arrow_length * dx, arrow_y - arrow_length * dy)
            else:
                arrow_start = (arrow_x - arrow_length * dx, arrow_y - arrow_length * dy)
                arrow_end = (arrow_x + arrow_length * dx, arrow_y + arrow_length * dy)

            arc = pat.Arc(
                center,
                2 * radius,
                2 * radius,
                angle=0,
                theta1=np.degrees(angle_start),
                theta2=np.degrees(angle_end),
                linewidth=edge_width,
                color=edge_color,
            )
            arrow = pat.FancyArrowPatch(
                arrow_start,
                arrow_end,
                color=arrow_color,
                arrowstyle="wedge",
                mutation_scale=arrow_width,
            )
            return (arc, arrow)
        else:
            start_x, start_y = angle_start
            end_x, end_y = angle_end
            (line,) = plt.plot(
                [start_x, end_x],
                [start_y, end_y],
                linewidth=edge_width,
                color=edge_color,
            )
            line.remove()

            arrow_vec = (end_x - start_x, end_y - start_y)
            arrow_vec_norm = np.linalg.norm(arrow_vec)
            arrow_x = start_x * (1 - arrow_pos) + end_x * arrow_pos
            arrow_y = start_y * (1 - arrow_pos) + end_y * arrow_pos

            dx = arrow_vec[0] / arrow_vec_norm
            dy = arrow_vec[1] / arrow_vec_norm

            arrow_start = (arrow_x - arrow_length * dx, arrow_y - arrow_length * dy)
            arrow_end = (arrow_x + arrow_length * dx, arrow_y + arrow_length * dy)

            arrow = pat.FancyArrowPatch(
                arrow_start,
                arrow_end,
                color=arrow_color,
                arrowstyle="wedge",
                mutation_scale=arrow_width,
            )

            return (line, arrow)


class NodeDrawer(Drawer):
    def __init__(self, ring_size: float):
        self.ring_size = ring_size
        self.font: dict[str, NodeFont] = {}

    def draw(self, node_type: SupportNodeType, angle_start, angle_end, face_width):
        return Drawer.draw_wedge(
            self.ring_size,
            angle_start,
            angle_end,
            face_width,
            **asdict(self.font[node_type])
        )

    def config_font(self, node_type: SupportNodeType, font: NodeFont) -> None:
        self.font[node_type] = font


class EdgeDrawer(Drawer):
    def __init__(self):
        self.font: dict[str, EdgeFont] = {}

    def draw(self, edge_type: SupportEdgeType, center, radius, angle_start, angle_end):
        return super().draw_arrow_arc(
            center, radius, angle_start, angle_end, **asdict(self.font[edge_type])
        )

    def config_font(self, edge_type: SupportEdgeType, font: EdgeFont):
        self.font[edge_type] = font


if __name__ == "__main__":
    fig, ax = plt.subplots()
    line, arrow = Drawer.draw_arrow_arc(
        (0, 0),
        0.3,
        [0, 1],
        [0, -1],
        1,
        "b",
        1,
        1,
        "b",
    )
    ax.add_artist(line)
    ax.add_artist(arrow)
    fig.show()
