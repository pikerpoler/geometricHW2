from matplotlib import pyplot as plt
import numpy as np
import math
from functools import lru_cache

timestep =0.01
epsilon = 0.00001





class Curve:
    def __init__(self, C, points):
        self.f = np.array([C(p) for p in points])
        self.points = np.array(points)
        self.calc_normals()
        self.calc_kappas()

    def calc_normals(self):
        normals = []
        tangents = []
        for i in range(len(self.points)):
            p_left= self.points[i-1]
            p_right = self.points[(i+1) % len(self.points)]
            dx = self.points[2] - self.points[0]

            tangent = (self.f[(i+1)% len(self.points)] - self.f[(i-1)% len(self.points)])/ dx
            tangents.append(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            normal /= np.linalg.norm(normal)
            normals.append(normal)
        self.normals = np.array(normals)
        self.tangents = np.array(tangents)

    def calc_kappas(self):
        kappas = []
        for i in range(len(self.tangents)):
            t_left = self.tangents[i-1]
            t_right = self.tangents[(i+1) % len(self.tangents)]
            dx = dx = self.points[2] - self.points[0]
            kappa = (self.tangents[(i+1)% len(self.tangents)] - self.tangents[(i-1)% len(self.points)])/ dx
            kappa = np.linalg.norm(kappa) ** (1/3)
            kappas.append([kappa, kappa])
        self.kappas = np.array(kappas)
        print(self.kappas[0], self.kappas[25])

    def __getitem__(self, item):
        return np.concatenate((self.f[:, item], self.f[[0], item] ))

    def __len__(self):
        return len(self.f)

    def do_step(self, section=2):
        if section == 2:
            self.f += timestep * self.normals
        elif section == 3:
            self.f += timestep * self.kappas * self.normals
        self.calc_normals()
        self.calc_kappas()


def section1():
    p = np.linspace(0, 2 * math.pi, 100)[: -1]
    C = lambda x: np.array([math.cos(x), math.sin(x)])
    curve = Curve(C, p)

    # make axes square
    # plt.axis('equal')
    plt.plot(curve[0], curve[1])
    plt.show()

def section(section=2):
    p = np.linspace(0, 2 * math.pi, 99)[:-1]
    # C = lambda x: np.array([math.cos(x), math.sin(x)])
    C = lambda x: np.array([2* math.cos(x), 4*math.sin(x)])
    curve = Curve(C, p)

    for t in range(100):
        curve.do_step(section=section)
        print(f't = {t}')
        if t % 1 == 0:
            # set axis to be of length 2
            plt.axis("equal")
            plt.plot(curve[0], curve[1])
            plt.title("t = " + str(t))
            plt.savefig(f'./images/{t}.png')
            plt.clf()

def section3():
    p = np.linspace(0, 2 * math.pi, 100)[:-1]
    # C = lambda x: np.array([math.cos(x), math.sin(x)])
    C = lambda x: np.array([2 * math.cos(x), math.sin(x)])
    curve = Curve(C, p)




if __name__ == '__main__':
    section1()
    # section(2)
    section(3)