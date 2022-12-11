from matplotlib import pyplot as plt
import numpy as np
import math
from functools import lru_cache

timestep =0.3
epsilon = 0.00001


def calculate_normal(p, C):
    t = (C(p + epsilon) - C(p-epsilon)) / (2 * epsilon)
    return np.array([-t[1], t[0]])

def get_C_func(t, C):
    return lambda x: C_t(x, t, C)


def C_t(x, t, C):
    if t == 0:
        return C(x)
    else:
        return C_t(x, t-1, C) + timestep * calculate_normal(x, get_C_func(t-1, C))

def section1():
    p = np.linspace(0, 2 * math.pi, 100)[: -1]
    C = lambda x: np.array([math.cos(x), math.sin(x)])
    curve = ([], [])
    for i in p:
        curve[0].append(C(i)[0])
        curve[1].append(C(i)[1])

    # make axes square
    plt.axis('equal')
    plt.plot(curve[0], curve[1])
    plt.show()

def section2():
    p = np.linspace(0, 2 * math.pi, 100)[: -1]
    C = lambda x,t=0: np.array([math.cos(x), math.sin(x)])
    curve = ([], [])
    for i in p:
        curve[0].append(C(i)[0])
        curve[1].append(C(i)[1])

    # make axes square
    plt.axis('equal')
    plt.plot(curve[0], curve[1])
    plt.show()
    for t in range(100):
        curve = ([], [])
        for i in p:
            curve[0].append(C_t(i, t, C)[0])
            curve[1].append(C_t(i, t, C)[1])
        plt.axis('equal')
        plt.plot(curve[0], curve[1])
        plt.title("t = " + str(t))
        plt.show()


if __name__ == '__main__':
    section1()
    section2()