"""
File for the Body class used in the n-body simulation
"""

from __future__ import annotations

import constants
import numpy as np

from misc import sq_len
from typing import Optional

# For typing
numeric = int | float


class Body:
    def __init__(self, m: numeric, v0: np.ndarray, pos: np.ndarray,
                 r: numeric) -> None:
        """
        :param m: Mass in units of Earth masses
        :param v0: Initial velocity of the body [km/s]
        :param pos: Position of the body at the start [km]
        :param r: Radius of the body [Earth radiuses]
        """
        self.m = m * constants.m_e
        self.vel = v0 * 1e3
        self.pos = pos * constants.au
        self.r = r * constants.r_e

    def get_pos(self) -> np.ndarray:
        """
        :return:
        """
        return self.pos

    def get_radius(self) -> numeric:
        """
        :return:
        """
        return self.r

    def get_mass(self) -> numeric:
        """
        :return:
        """
        return self.m

    def get_vel(self) -> np.ndarray:
        """
        :return:
        """
        return self.vel

    def update(self, acc: np.ndarray, dt: numeric) -> None:
        """
        Updates the position of the body using the "drift-kick-drift"
        integration method
        :param acc:
        :param dt:
        :return:
        """
        self.vel += acc * dt * 0.5
        self.pos += self.vel * dt
        self.vel += acc * dt * 0.5

    def check_collision(self, other: Body) -> bool:
        """
        :param other:
        :return:
        """
        direc = other.get_pos() - self.get_pos()
        r_sum = self.get_radius() + other.get_radius()
        if sq_len(direc) <= (r_sum * r_sum):
            return True
        return False

    def _end_velocity(self, other: Body) -> np.ndarray:
        """
        End velocity for the combined bodies after a perfectly inelastic
        collision
        :param other:
        :return:
        """
        m1, m2 = self.m, other.m
        v1, v2 = self.get_vel(), other.get_vel()
        return (m1 * v1 + m2 * v2) / (m1 + m2)

    def combine_bodies(self, other: Body) -> Optional[Body]:
        """
        :param other:
        :return:
        """
        m = (self.get_mass() + other.get_mass()) / constants.m_e
        pos = self.get_pos() if self.get_mass() > other.get_mass() else other.get_pos()
        pos /= constants.au
        vel = self._end_velocity(other) * 1e-3
        r1 = self.get_radius()
        r2 = other.get_radius()
        r = np.sqrt(r1 * r1 + r2 * r2) / constants.r_e
        return Body(m=m, v0=vel, pos=pos, r=r)

    def __repr__(self) -> str:
        """
        :return:
        """
        return f'Body at x={self.pos[0]}, y={self.pos[1]}'

    def __str__(self) -> str:
        """
        :return:
        """
        return f'Body at x={self.pos[0]}, y={self.pos[1]}'
