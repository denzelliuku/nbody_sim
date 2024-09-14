"""
File for the Body class used in the n-body simulation
"""

from __future__ import annotations

import constants
import numpy as np

from misc import sq_len
from typing import Optional


class Body:
    def __init__(self, m: int | float, v0: np.ndarray, pos: np.ndarray,
                 r: int | float, acc: np.ndarray = None) -> None:
        """
        :param m: Mass in units of Earth masses
        :param v0: Initial velocity of the body [km/s]
        :param pos: Position of the body at the start [km]
        :param r: Radius of the body [Earth radiuses]
        :param acc: Initial acceleration of the body [m/s^2]
        """
        self.m = m * constants.m_e
        self.vel = v0 * 1e3
        self.pos = pos * constants.au
        self.r = r * constants.r_e
        if acc is None:
            self.acc = np.zeros((2, ))
        else:
            self.acc = acc

    def update(self, acc: np.ndarray, dt: int | float) -> None:
        """
        Updates the position of the body using the "kick-drift-kick"
        integration method
        :param acc: Acceleration at the current timestep
        :param dt: Size of the timestep
        :return:
        """
        # Update the velocity using the acceleration at the previous timestep
        self.vel += self.acc * dt * 0.5

        # Update the position using the velocity updated with the acceleration
        # at the previous timestep
        self.pos += self.vel * dt

        # Update the velocity a second time with the acceleration of the current
        # timestep
        self.vel += acc * dt * 0.5

        # Save the acceleration for use at the next time step
        self.acc = acc

    def check_collision(self, other: Body) -> bool:
        """
        :param other:
        :return:
        """
        direc = other.pos - self.pos
        r_sum = self.r + other.r
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
        v1, v2 = self.vel, other.vel
        return (m1 * v1 + m2 * v2) / (m1 + m2)

    def combine_bodies(self, other: Body) -> Optional[Body]:
        """
        :param other:
        :return:
        """
        m = (self.m + other.m) / constants.m_e
        pos = self.pos if self.m > other.m else other.pos
        pos /= constants.au
        vel = self._end_velocity(other) * 1e-3
        r1 = self.r
        r2 = other.r
        r = np.sqrt(r1 * r1 + r2 * r2) / constants.r_e
        return Body(m=m, v0=vel, pos=pos, r=r)

    def __repr__(self) -> str:
        """
        :return:
        """
        return f"Body at x={self.pos[0]}, y={self.pos[1]}"

    def __str__(self) -> str:
        """
        :return:
        """
        return f"Body at x={self.pos[0]}, y={self.pos[1]}"

