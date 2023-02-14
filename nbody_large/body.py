"""
File for the Body class and System class used in the n-body simulation
"""

from __future__ import annotations

import constants
import numpy as np

from copy import copy
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

    def _acc(self, others: list, epsilon: numeric) -> np.array:
        """
        Calculates the gracitational acceleration between the two bodies
        :param others: List of the other bodies
        :param epsilon: A number added to the distances between the bodies
        so that the acceleration doesn't become infinite when the distance
        is (close to) zero
        :return:
        """
        acc = np.zeros(2)
        for b in others:
            direc = b.get_pos() - self.get_pos()
            angle = np.arctan2(direc[1], direc[0])
            denom = np.power(sq_len(direc) + epsilon * epsilon, 3 / 2)
            mag = constants.big_g * b.get_mass() * np.sqrt(sq_len(direc)) / denom
            acc += np.array([np.cos(angle), np.sin(angle)]) * mag
        return acc

    def update_position(self, others: list, dt: numeric, epsilon: numeric) -> None:
        """
        Updates the position of the body due to the gravitational attraction
        of the other body
        :param others:
        :param dt:
        :param epsilon: A number added to the distances between the bodies
        so that the acceleration doesn't become infinite when the distance
        is (close to) zero
        :return:
        """
        acc = self._acc(others, epsilon)
        self.pos += self.vel * dt + 0.5 * acc * dt * dt
        self.vel += acc * dt

    def new_update(self, acc: np.ndarray, dt: numeric) -> None:
        """
        Updates the position of the body
        :param acc:
        :param dt:
        :return:
        """
        self.pos += self.vel * dt + 0.5 * acc * dt * dt
        self.vel += acc * dt

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


class System:
    def __init__(self, epsilon: numeric = 1e2) -> None:
        """
        :param epsilon: A number added to the distances between the bodies
        so that the acceleration doesn't become infinite when the distance
        is (close to) zero
        """
        self._bodies: list[Body] = []
        self.epsilon = epsilon

    def add_body(self, body: Body) -> None:
        """
        :param body:
        :return:
        """
        self._bodies.append(body)

    def remove_body(self, body: Body) -> None:
        """
        :param body:
        :return:
        """
        self._bodies.remove(body)

    def get_bodies(self) -> list:
        """
        :return:
        """
        return self._bodies

    def step_forward(self, dt: numeric) -> None:
        """
        :param dt:
        :return:
        """
        for b in self._bodies:
            others = copy(self._bodies)
            others.remove(b)
            b.update_position(others, dt, self.epsilon)
