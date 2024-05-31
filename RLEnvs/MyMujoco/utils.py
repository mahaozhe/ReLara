import numpy as np


def gen_rand_point_within(center, min_radius, max_radius):
    """generate a random point within a concentric circle"""
    radius = np.random.uniform(min_radius, max_radius)

    angle = np.random.uniform(0, np.pi * 2)

    x = center[0] + radius * np.cos(angle)
    y = center[1] + radius * np.sin(angle)

    return x, y


def gen_rand_num_within(a, b):
    """generate a random number within [a, b] union [-b, -a]"""
    r = np.random.uniform(a, b)
    if np.random.random() < 0.5:
        r = -r
    return r
