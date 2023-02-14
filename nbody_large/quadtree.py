"""
Quadtree data structure that is used to split the computational domain of the
n-body system into smaller areas
"""

import constants
import numpy as np

from body import Body
from misc import vector_len

# For typing
numeric = int | float


class Node:
    """
    A node (quadrant) in the quadtree
    """
    def __init__(self, pos: np.ndarray, w: numeric, h: numeric,
                 bodies: list[Body]) -> None:
        """
        :param pos: Location of the middle point of the node
        :param w: Width of the node
        :param h: Height of the node
        :param bodies: List of bodies contained by the node
        """
        self.pos = pos
        self.w = w
        self.h = h
        self._bodies = bodies
        self.cm = np.zeros(shape=(2, ))
        self.total_mass = sum(b.get_mass() for b in bodies)
        self.children = []
        self._create_child_nodes()

    def _create_node(self, quad: str, bodies: list) -> None:
        """
        Assigns the given list of bodies into the appropriate quadrant, and creates
        the child node
        :param quad:
        :param bodies:
        :return:
        """
        half_w, half_h = self.w / 2, self.h / 2
        if quad == 'nw':
            x, y = self.pos[0] - half_w, self.pos[1] + half_h
            node_nw = Node(pos=np.array([x, y]), bodies=bodies, w=half_w, h=half_h)
            self.children.append(node_nw)
        elif quad == 'ne':
            x, y = self.pos[0] + half_w, self.pos[1] + half_h
            node_ne = Node(pos=np.array([x, y]), bodies=bodies, w=half_w, h=half_h)
            self.children.append(node_ne)
        elif quad == 'sw':
            x, y = self.pos[0] - half_w, self.pos[1] - half_h
            node_sw = Node(pos=np.array([x, y]), bodies=bodies, w=half_w, h=half_h)
            self.children.append(node_sw)
        elif quad == 'se':
            x, y = self.pos[0] + half_w, self.pos[1] - half_h
            node_se = Node(pos=np.array([x, y]), bodies=bodies, w=half_w, h=half_h)
            self.children.append(node_se)
        else:
            print('Uncreachable sheit')
            quit(-1)

    def _create_child_nodes(self) -> None:
        """
        Creates the child nodes if necessary
        :return:
        """
        # If this node doesn't contain any bodies, nothing needs to be done
        if not self._bodies:
            return
        # If there's only one body, calculate the center of mass but don't create
        # any child nodes
        if len(self._bodies) == 1:
            self.cm = self._bodies[0].get_pos()
            return
        # Create the child nodes for quadrants that have bodies in them
        b_quads = self._split_bodies()
        self._calc_cm()
        for quad in b_quads.keys():
            self._create_node(quad, b_quads[quad])

    @staticmethod
    def _find_quadrant(b_pos: np.ndarray, q_pos: np.ndarray) -> str:
        """
        Returns the quadrant that the body belongs to
        :param b_pos:
        :param q_pos:
        :return:
        """
        vec = b_pos - q_pos
        direc = np.arctan2(vec[1], vec[0])
        pi2 = np.pi * 0.5
        if 0 < direc <= pi2:
            return 'ne'
        elif pi2 < direc <= np.pi:
            return 'nw'
        elif -pi2 < direc <= 0:
            return 'sw'
        elif -np.pi < direc <= -pi2:
            return 'se'
        print('Unreachable quadrant')
        quit(-1)

    def _split_bodies(self) -> dict:
        """
        Splits the bodies into the child nodes of this node
        :return:
        """
        b_quads = {'nw': [], 'ne': [], 'sw': [], 'se': []}
        for b in self._bodies:
            quad = self._find_quadrant(b.get_pos(), self.pos)
            b_quads[quad].append(b)
        return b_quads

    def _calc_cm(self) -> None:
        """
        Calculates the center of the mass of the node
        :return:
        """
        prod = sum(b.get_mass() * b.get_pos() for b in self._bodies)
        self.cm = 1 / self.total_mass * prod


class QuadTree:
    def __init__(self, bodies: list[Body], dt: numeric, limit: numeric = 5,
                 eps: numeric = 1e3) -> None:
        self.dt = dt
        self.limit = limit
        self.eps = eps
        pos = np.array([0., 0.])
        self._bodies = bodies
        dist = max(vector_len(b.get_pos()) for b in bodies)
        self.root = Node(pos=pos, w=dist, h=dist, bodies=bodies)

    def get_bodies(self) -> list:
        """
        :return:
        """
        return self._bodies

    def step_forward(self) -> None:
        """
        Takes one step forward in the simulation, i.e., computes the
        next positions of the bodies
        :return:
        """
        # Calculate the acceleration for each body
        for body in self._bodies:
            stack = [self.root]
            acc = np.zeros(shape=(2, ))
            while stack:
                node = stack.pop()
                size = np.mean((node.w, node.h))
                vec = node.cm - body.get_pos()
                dist = vector_len(vec)
                if dist == 0:
                    continue
                if (size / dist) < self.limit:
                    angle = np.arctan2(vec[1], vec[0])
                    denom = np.power(dist * dist + self.eps * self.eps, 3 / 2)
                    mag = constants.big_g * node.total_mass * dist / denom
                    acc += np.array([np.cos(angle), np.sin(angle)]) * mag
                    continue
                for child in node.children:
                    stack.append(child)

            # Update the position of the body
            body.new_update(acc=acc, dt=self.dt)
