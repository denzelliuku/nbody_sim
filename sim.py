"""
A more optimised version of the N-body simulation that uses a quadtree structure
to decrease the time complexity of the simulation
"""

import pygame
import constants
import numpy as np

from body import Body
from misc import vector_len
from quadtree import QuadTree
from timeit import default_timer


def _next_magnitude(n: int | float) -> int | float:
    """
    Returns the next power of ten from n
    :param n:
    :return:
    """
    return np.power(10, np.ceil(np.log10(n)))


def _scale_position(bodies: list[Body]) -> int | float:
    """
    Returns the scale so that the all bodies can be seen on the screen
    :param bodies:
    :return:
    """
    max_pos = max(vector_len(b.pos) for b in bodies)
    return _next_magnitude(max_pos)


def _determine_radii(qtree: QuadTree, body: Body, rad_range: tuple) -> int:
    """
    Determines the radii of the body in pixels
    :param qtree:
    :param body:
    :param rad_range:
    :return:
    """
    min_r, max_r = rad_range
    max_m = max(b.m for b in qtree.bodies)
    return max(min_r, int(body.m / max_m * max_r))


def _calc_bar_value(w: int, bar_len: int, pos_scale: int | float) -> int | float:
    """
    Calculates the distance (in appropriate units) that the length of the
    scale bar (in pixels) corresponds to
    :param w: Width of the image in pixels
    :param bar_len: Length of the scale bar in pixels
    :param pos_scale: The value used to scale the bodies positions in meters
    :return: The value and its unit
    """
    len_per_pixel = pos_scale / w
    return bar_len * len_per_pixel / constants.au


def _scale_time(t: int | float) -> tuple[int | float, str]:
    """
    :param t: Time in seconds
    :return: Value of scaled time and the appropriate unit for the
    time as a string
    """
    if t < 3600:
        return t / 60, "mins"
    if t < 7 * 24 * 3600:
        return t / 3600, "hours"
    return t / (24 * 3600), "days"


def animate(bodies: list[Body], dt: int | float, limit: int | float,
            eps: int | float) -> None:
    """
    :param bodies:
    :param dt:
    :param limit:
    :param eps:
    :return:
    """
    bg_color = (0, 0, 0)  # Background color
    body_color = (0, 255, 0)  # Color for the bodies
    font_color = (255, 255, 255)  # Color for the texts
    padx, pady = 10, 10  # Pixels
    font = pygame.font.SysFont("arial", 13)
    w, h = 800, 600  # Pixels
    window = pygame.display.set_mode((w, h))
    pos_scale = _scale_position(bodies=bodies)  # [m]
    scale_bar_len = w // 16  # Pixels
    min_r, max_r = 1, 5  # Pixels
    start = default_timer()  # To track fps
    elapsed = 0  # To track the total elapsed time
    dragging = False
    m_x0, m_y0 = 0, 0  # Initial position of the mouse when dragging
    dx, dy = 0, 0  # Change in the position of the mouse while dragging
    offset_x, offset_y = w // 2, h // 2  # Shift the origin of the coordinates
    run = True
    while run:
        window.fill(bg_color)
        # Create the QuadTree for this timestep
        qtree = QuadTree(bodies=bodies, dt=dt, limit=limit, eps=eps)
        # Handle input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.MOUSEWHEEL:
                factor = np.power(10, np.floor(np.log10(pos_scale) - 1))
                if event.y < 0:
                    pos_scale += 5 * factor
                else:
                    pos_scale -= 5 * factor
            elif event.type == pygame.MOUSEBUTTONDOWN:
                m_x0, m_y0 = pygame.mouse.get_pos()
                dragging = True
            elif event.type == pygame.MOUSEBUTTONUP:
                offset_x += dx
                offset_y += dy
                dx, dy = 0, 0
                dragging = False

        # Update the state of the bodies
        qtree.step_forward()

        # Render the new state
        if dragging:
            m_x1, m_y1 = pygame.mouse.get_pos()
            dx = m_x1 - m_x0
            dy = m_y1 - m_y0
        for body in qtree.bodies:
            x, y = body.pos
            x = x / pos_scale * w + offset_x + dx
            y = y / pos_scale * h + offset_y + dy
            r = _determine_radii(qtree, body, (min_r, max_r))
            pygame.draw.circle(window, body_color, (x, y), r)

        # Render the scaling bar showing the distance scale
        bar_x, bar_y = w - scale_bar_len - padx, h - (pady * 5)
        pygame.draw.line(window, font_color, (bar_x, bar_y),
                         (bar_x + scale_bar_len, bar_y))
        bar_value = _calc_bar_value(w, scale_bar_len, pos_scale)
        bar_text = font.render(f"{bar_value:.3f} AU", True, font_color)
        bar_text_x, bar_text_y = bar_x + scale_bar_len / 6, bar_y + 2
        window.blit(bar_text, (bar_text_x, bar_text_y))

        # Render fps
        end = default_timer()
        fps = 1 / (end - start)
        start = end
        fps_text = font.render(f"fps: {fps:.1f}", True, font_color)
        fps_x, fps_y = w - fps_text.get_width() - padx, pady / 2
        window.blit(fps_text, (fps_x, fps_y))

        # Render elapsed time
        elapsed += qtree.dt  # Elapsed time in seconds
        scaled_time, time_unit = _scale_time(elapsed)
        time_text = font.render(f"Duration: {scaled_time:.1f} {time_unit}",
                                True, font_color)
        time_x = w - time_text.get_width() - padx
        time_y = fps_y + time_text.get_height() / 2 + pady
        window.blit(time_text, (time_x, time_y))

        # Render the number of bodies left
        n_bodies = len(qtree.bodies)
        bodies_text = font.render(f"Bodies: {n_bodies}", True, font_color)
        bodies_x = w - bodies_text.get_width() - padx
        bodies_y = time_y + bodies_text.get_height() / 2 + pady
        window.blit(bodies_text, (bodies_x, bodies_y))

        pygame.display.update()
    pygame.display.quit()


def _randrange(a: int | float, b: int | float) -> float:
    """
    :param a:
    :param b:
    :return:
    """
    return np.random.random() * (b - a) + a


def solar_system() -> list[Body]:
    """
    The planets and the Sun as Bodies in a list
    :return:
    """
    b1 = Body(m=332950, v0=np.array([0., 0.]), pos=np.array([0, 0.1]), r=109)  # Sun
    b2 = Body(m=0.055, v0=np.array([0, -47.36]), pos=np.array([.387098, 0.2]),
              r=.3829)  # Mercury
    b3 = Body(m=.815, v0=np.array([0, -35.02]), pos=np.array([.723332, 0.3]),
              r=.9499)  # Venus
    b4 = Body(m=1, v0=np.array([0, -29.78]), pos=np.array([1, 0.4]), r=1)  # Earth
    b5 = Body(m=.107, v0=np.array([0, -24.07]), pos=np.array([1.52368055, 0.5]),
              r=0.532)  # Mars
    b6 = Body(m=317.8, v0=np.array([0, -13.07]), pos=np.array([5.2038, 0.6]),
              r=10.973)  # Jupiter
    b7 = Body(m=95.159, v0=np.array([0, -9.68]), pos=np.array([9.5826, 0.7]),
              r=9.1402)  # Saturnus
    b8 = Body(m=14.536, v0=np.array([0, -6.8]), pos=np.array([19.19126, 0.8]),
              r=4)  # Uranus
    b9 = Body(m=17.147, v0=np.array([0, -5.43]), pos=np.array([30.07, 0.9]),
              r=3.85)  # Neptune
    return [b1, b2, b3, b4, b5, b6, b7, b8, b9]


def random_system(n_bodies: int, mass_range: tuple, vel_range: tuple,
                  pos_range: tuple, rad_range: tuple) -> list[Body]:
    """
    Returns a list of Bodies with random properties (within the given
    ranges)
    :param n_bodies:
    :param mass_range:
    :param vel_range:
    :param pos_range:
    :param rad_range:
    :return:
    """
    min_mass, max_mass = mass_range
    min_vel, max_vel = vel_range
    min_pos, max_pos = pos_range
    min_r, max_r = rad_range
    bodies = []
    for _ in range(n_bodies):
        mass = _randrange(min_mass, max_mass)
        r = _randrange(min_r, max_r)
        vel_angle = _randrange(0, 2 * np.pi)
        vel_direc = np.array([np.cos(vel_angle), np.sin(vel_angle)])
        vel_mag = _randrange(min_vel, max_vel)
        vel = vel_direc * vel_mag
        pos_angle = _randrange(0, 2 * np.pi)
        pos_direc = np.array([np.cos(pos_angle), np.sin(pos_angle)])
        dist = _randrange(min_pos, max_pos)
        pos = pos_direc * dist
        body = Body(m=mass, v0=vel, pos=pos, r=r)
        bodies.append(body)
    return bodies


def main() -> None:
    """
    :return:
    """
    pygame.font.init()
    dt = 1 / 5 * 3600  # [s]
    limit = 0
    eps = 1e1
    n_bodies = 100
    mass_range = (0.1, 318)  # Earth masses
    vel_range = (1, 10)  # [km/s]
    pos_range = (0, .1)  # [AU]
    rad_range = (0.1, 11)  # Earth radiuses
    bodies = random_system(n_bodies=n_bodies, mass_range=mass_range, vel_range=vel_range,
                           pos_range=pos_range, rad_range=rad_range)
    animate(bodies=bodies, dt=dt, limit=limit, eps=eps)


if __name__ == "__main__":
    main()

