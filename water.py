import numpy as np
from itertools import product
from vxl_importer import import_vxl

frictionFactor = 0.5
g = 10
dx = dy = 10
dt = 1

class Simulator:
    def __init__(self, vxl:str):
        self.terrain, self.water = import_vxl(vxl)
        self.N = self.terrain.shape[0]
        self.flowx = np.zeros([self.N + 1, self.N])
        self.flowy = np.zeros([self.N, self.N + 1])

    def step(self):
        self.flowx[0, :] = 0
        self.flowx[self.N, :] = 0
        self.flowy[:, 0] = 0
        self.flowy[:, self.N] = 0

        flow_x_temp = np.copy(self.flowx)
        flow_y_temp = np.copy(self.flowy)

        f_order = list(product(range(1, self.N), range(self.N)))
        for x, y in f_order:
            boost = 0
            if self.flowx[x, y] < 0:
                boost = -flow_x_temp[x + 1, y] / max(self.water[x,y], 0.001)
            elif self.flowx[x, y] > 0:
                boost = flow_x_temp[x - 1, y] / max(self.water[x,y], 0.001)
            boost = max(0, min(1, boost))
            boost = np.exp(boost)

            self.flowx[x, y] = flow_x_temp[x, y] * frictionFactor + \
                          boost * ((self.water[x - 1, y] + self.terrain[x - 1, y]) - (self.water[x, y] + self.terrain[x, y])) * g * dt / dx

        for y, x in f_order:
            boost = 0
            if self.flowy[x, y] < 0:
                boost = -flow_y_temp[x, y + 1] / max(self.water[x,y], 0.001)
            elif self.flowy[x, y] > 0:
                boost = flow_y_temp[x, y - 1] / max(self.water[x,y], 0.001)
            boost = max(0, min(1, boost))
            boost = np.exp(boost)

            self.flowy[x, y] = flow_y_temp[x, y] * frictionFactor + \
                          boost * ((self.water[x, y - 1] + self.terrain[x, y - 1]) - (self.water[x, y] + self.terrain[x, y])) * g * dt / dx

        scale_order = list(product(range(self.N), range(self.N)))
        np.random.shuffle(scale_order)
        for y, x in scale_order:
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
            for x in range(self.N):
                self.water[x, y] += (self.flowx[x, y] + self.flowy[x, y] - self.flowx[x + 1, y] - self.flowy[x, y + 1]) * dt / dx / dy