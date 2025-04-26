import numpy as np
from vxl_importer import import_vxl

frictionFactor = 0.5
g = 10
dx = dy = 10
dt = 1

class Simulator:
    def __init__(self, vxl:str):
        self.terrain, self.water = import_vxl(r'data\lake_fill.txt')
        self.N = self.terrain.shape[0]
        self.flowx = np.zeros([self.N + 1, self.N])
        self.flowy = np.zeros([self.N, self.N + 1])
        self.vx = np.zeros([self.N + 1, self.N])
        self.vy = np.zeros([self.N, self.N + 1])

    def step(self):
        self.flowx[0, :] = 0
        self.flowx[self.N, :] = 0
        self.flowy[:, 0] = 0
        self.flowy[:, self.N] = 0

        for y in range(self.N):
            for x in range(1, self.N):
                self.flowx[x, y] = self.flowx[x, y] * frictionFactor + \
                              ((self.water[x - 1, y] + self.terrain[x - 1, y]) - (self.water[x, y] + self.terrain[x, y])) * g * dt / dx

        for y in range(1, self.N):
            for x in range(self.N):
                self.flowy[x, y] = self.flowy[x, y] * frictionFactor + \
                              ((self.water[x, y - 1] + self.terrain[x, y - 1]) - (self.water[x, y] + self.terrain[x, y])) * g * dt / dx

        for y in range(self.N):
            for x in range(self.N):
                total_outflow = 0
                total_outflow += max(0, -self.flowx[x, y])
                total_outflow += max(0, -self.flowy[x, y])
                total_outflow += max(0, self.flowx[x + 1, y])
                total_outflow += max(0, self.flowy[x, y + 1])

                if total_outflow > 0:
                    max_outflow = self.water[x, y] * dx * dy / dt
                    scale = min(1, max_outflow / total_outflow)
                    if self.flowx[x, y] < 0:
                        self.flowx[x, y] *= scale
                    if self.flowx[x + 1, y] > 0:
                        self.flowx[x + 1, y] *= scale
                    if self.flowy[x, y] < 0:
                        self.flowy[x, y] *= scale
                    if self.flowy[x, y + 1] > 0:
                        self.flowy[x, y + 1] *= scale

        for y in range(self.N):
            for x in range(1, self.N):
                self.vx[x, y] = self.flowx[x, y] / max(self.water[x-1, y], self.water[x, y], 0.001)
        for y in range(self.N):
            for x in range(1, self.N):
                self.vy[x, y] = self.flowy[x, y] / max(self.water[x, y-1], self.water[x, y], 0.001)

        for y in range(self.N):
            for x in range(self.N):
                self.water[x, y] += (self.flowx[x, y] + self.flowy[x, y] - self.flowx[x + 1, y] - self.flowy[x, y + 1]) * dt / dx / dy