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

        boost_x1 = np.maximum(0, -np.sign(self.flowx[1:-1,:])) * (-self.flowx[2:,:]) / np.maximum(self.water[1:,:], 0.001)
        boost_x2 = np.maximum(0, np.sign(self.flowx[1:-1,:])) * self.flowx[:-2,:] / np.maximum(self.water[:-1,:], 0.001)
        boost_x = np.maximum(boost_x1, boost_x2)
        boost_x = np.minimum(1, boost_x)
        boost_x = np.exp(boost_x)
        self.flowx[1:-1] = self.flowx[1:-1] * frictionFactor + \
            boost_x * ((self.water[:-1,:] + self.terrain[:-1:]) - (self.water[1:,:] + self.terrain[1:,:])) * g * dt / dx

        boost_y1 = np.maximum(0, -np.sign(self.flowy[:,1:-1])) * (-self.flowy[:,2:]) / np.maximum(self.water[:,1:], 0.001)
        boost_y2 = np.maximum(0, np.sign(self.flowy[:,1:-1])) * self.flowy[:,:-2] / np.maximum(self.water[:,:-1], 0.001)
        boost_y = np.maximum(boost_y1, boost_y2)
        boost_y = np.minimum(1, boost_y)
        boost_y = np.exp(boost_y)
        self.flowy[:,1:-1] = self.flowy[:,1:-1] * frictionFactor + \
            boost_y * ((self.water[:,:-1] + self.terrain[:,:-1]) - (self.water[:,1:] + self.terrain[:,1:])) * g * dt / dx

        # Non-parallel part
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

        self.water += (self.flowx[:-1] + self.flowy[:,:-1] - self.flowx[1:] - self.flowy[:,1:]) * dt / dx / dy
