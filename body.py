"""
File for the Body class used in the n-body simulation
"""

from __future__ import annotations

import constants
import numpy as np

from misc import sq_len
from typing import Optional

TRACE_POSITIONS = 100
TRACE_INTERVAL = 20


def _conv_helper(n: int | float, h: int | float, s: int | float, v: int | float) \
    -> int | float:
    """
    Helper function used in the color conversion
    :param n:
    :param h:
    :param s:
    :param v:
    :return:
    """
    k = (n + h / 60) % 6
    return v - v * s * max(0, min(k, 4 - k, 1))


def _hsv2rgb(hue: int | float, saturation: int | float, value: int | float) \
    -> tuple[int, int, int]:
    """
    Conversion from wikipedia:
    https://en.wikipedia.org/wiki/HSL_and_HSV
    :param hue:
    :param saturation:
    :param value:
    :return:
    """
    r = _conv_helper(n=5, h=hue, s=saturation, v=value) * 255
    g = _conv_helper(n=3, h=hue, s=saturation, v=value) * 255
    b = _conv_helper(n=1, h=hue, s=saturation, v=value) * 255
    return int(r), int(g), int(b)


def _gen_random_color() -> tuple[int, int, int]:
    """
    :return:
    """
    hue = int(np.random.random() * 360)
    sat = 1
    val = 1
    return _hsv2rgb(hue=hue, saturation=sat, value=val)


class Body:
    def __init__(self, m: int | float, v0: np.ndarray, pos: np.ndarray,
                 r: int | float, acc: np.ndarray = None,
                 color: tuple[int, int, int] = None) -> None:
        """
        :param m: Mass in units of Earth masses
        :param v0: Initial velocity of the body [km/s]
        :param pos: Position of the body at the start [km]
        :param r: Radius of the body [Earth radiuses]
        :param acc: Initial acceleration of the body [m/s^2]
        :param color: Color that the body will be rendered in RGB format. If None,
        a random color will be generated.
        :return:
        """
        self.m = m * constants.m_e
        self.vel = v0 * 1e3
        self.pos = pos * constants.au
        self.trace = [self.pos.copy()]  # To keep track of older positions
        self.r = r * constants.r_e
        if acc is None:
            self.acc = np.zeros((2, ))
        else:
            self.acc = acc
        if color is None:
            self.color = _gen_random_color()
        else:
            self.color = color

    def _save_pos(self) -> None:
        """
        :return:
        """
        if len(self.trace) >= TRACE_POSITIONS:
            del self.trace[0]
            self.trace.append(self.pos.copy())
            return
        self.trace.append(self.pos.copy()) 

    def update(self, acc: np.ndarray, dt: int | float, n: int) -> None:
        """
        Updates the position of the body using the "kick-drift-kick"
        integration method
        :param acc: Acceleration at the current timestep
        :param dt: Size of the timestep
        :param n: Frame count
        :return:
        """
        # Update the velocity using the acceleration at the previous timestep
        self.vel += self.acc * dt * 0.5

        # Update the position using the velocity updated with the acceleration
        # at the previous timestep
        self.pos += self.vel * dt
        # Save the current position (at specific intervals)
        if n % TRACE_INTERVAL == 0:
            self._save_pos()

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
        return Body(m=m, v0=vel, pos=pos, r=r, color=self.color)

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

