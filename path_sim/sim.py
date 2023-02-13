"""
Simulation of circular motion between two gravitally attracted
objects with variable properties. One object is stationary, and the
other is in orbit around the stationary object.
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection

# For typing
numeric = int | float


def g_acc(m: numeric, r: numeric, theta: numeric) -> np.ndarray:
    """
    Calculates the acceleration due to gravity between the two objects
    :param m: Mass of the stationary object
    :param r: Distance between the objects
    :param theta: Angle between the distance vector of the objects
    and the positive x-axis
    :return: Acceleration as a vector in cartesian coordinates
    """
    mag = -6.67259e-11 * m / (r * r)
    return np.array([np.cos(theta), np.sin(theta)]) * mag


def simulate(tspan: np.ndarray, m: numeric, r: numeric,
             theta: numeric, v0: np.ndarray, r_min: numeric = None) \
        -> np.ndarray:
    """
    Calculates the location of the orbiting projectile at the given
    timepoints
    :param tspan: Timepoints in units of days
    :param m: Mass of the stationary object [kg]
    :param r: Distance between the objects [m]
    :param theta: Angle between the objects' distance vector and
    the positive x-axis at t == 0 [deg]
    :param v0: Initial velocity of the object as a vector in a
    cartesian coordinate system [m/s]
    :param r_min: Minimal distance between the objects before they
    crash in to each other, i.e. the sum of the radii of the objects
    :return:
    """
    dt = (tspan[1] - tspan[0]) * 24 * 3600
    vel = v0
    theta = np.deg2rad(theta)
    sol = np.zeros((1, 2))
    sol[0, 0] = r * np.cos(theta)
    sol[0, 1] = r * np.sin(theta)
    x, y = sol[0, 0], sol[0, 1]
    for i, _ in enumerate(tspan):
        acc_g = g_acc(m, r, theta)
        vel = vel + acc_g * dt
        dp = vel * dt
        x += dp[0]
        y += dp[1]
        r = np.sqrt((x * x) + (y * y))
        theta = np.arctan2(y, x)
        sol = np.vstack((sol, np.array([x, y])))
        if r_min is not None and r <= r_min:
            print(f'Crash after {i} days')
            break
    return sol


def seg_plot(data: np.ndarray, color: tuple = (0.5, 0, 0.5)) -> None:
    """
    Plots the trajectory with a changing transparency using line
    segments
    :param data: Array of x- and y-coordinates
    :param color: Color of the line as an RGB-value
    (default color is purple)
    :return:
    """
    alphas = np.linspace(0.1, 1, data.shape[0])
    points = np.reshape(data, (-1, 1, 2))
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    rgba = np.zeros((data.shape[0], 4))
    rgba[:, 0:3] = color
    rgba[:, 3] = alphas
    lc = LineCollection(segs, linewidths=3, colors=rgba)
    _, a = plt.subplots()
    plt.scatter(0, 0, c='r', s=100)
    plt.grid()
    a.add_collection(lc)
    plt.autoscale()
    plt.show()


def main():
    # Earth: 5.974e24, sun: 1.989e30
    m = 1.989e30
    d = 149.6e9
    # r_earth, r_sun = 6.37814e6, 6.96e8
    # r_min = r_earth + r_sun
    tspan = np.arange(0, 365, 1)  # Days
    theta = np.deg2rad(0)
    v_mag = 29780  # For Earth around sun: 29780 m/s
    offset = np.pi / 2
    v0 = np.array([np.cos(theta + offset), np.sin(theta + offset)]) * v_mag
    data = simulate(tspan, m, d, theta, v0)
    seg_plot(data)


if __name__ == '__main__':
    main()
