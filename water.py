import numpy as np
from itertools import product
from vxl_importer import import_vxl

frictionFactor = 0.02
g = 0.1
dx = 1
dt = 0.5
sourceRate = 0.2 # m/s

class Simulator:
    def __init__(self, vxl:str):
        terrain, water, source = import_vxl(vxl)
        self.M = terrain.shape[0] + 2
        self.N = terrain.shape[1] + 2

        self.terrain = np.empty((self.M,self.N))
        self.terrain[1:-1,1:-1] = terrain
        self.terrain[0,:] = self.terrain[1,:]
        self.terrain[-1,:] = self.terrain[-2,:]
        self.terrain[:,0] = self.terrain[:,1]
        self.terrain[:,-1] = self.terrain[:,-2]
        self.terrain[0,0] = self.terrain[0,-1] = self.terrain[-1,0] = self.terrain[-1,-1] = -1

        self.water = np.zeros((self.M,self.N))  # m
        self.water[1:-1,1:-1] = water

        self.source = {}
        for (x, y), intensity in source.items():
            self.source[(x+1,y+1)] = intensity

        self.flowx = np.zeros([self.M + 1, self.N]) #m^3/s
        self.flowy = np.zeros([self.M, self.N + 1])

    def step(self):
        for (x, y), intensity in self.source.items():
            self.water[x,y] += intensity * dt * sourceRate

        # boost causes varying result when scaling in length
        boost_x1 = np.maximum(0, -np.sign(self.flowx[1:-1,:])) * (-self.flowx[2:,:]) / np.maximum(self.water[1:,:], 0.001) # m^2/s
        boost_x2 = np.maximum(0, np.sign(self.flowx[1:-1,:])) * self.flowx[:-2,:] / np.maximum(self.water[:-1,:], 0.001)
        boost_x = np.maximum(boost_x1, boost_x2) / dx
        boost_x = np.minimum(1, boost_x)
        boost_x = np.exp(boost_x)
        self.flowx[1:-1] = self.flowx[1:-1] * (1 - frictionFactor * dt) + \
                           boost_x * ((self.water[:-1,:] + self.terrain[:-1:]) - (self.water[1:,:] + self.terrain[1:,:])) * g * dt * dx

        boost_y1 = np.maximum(0, -np.sign(self.flowy[:,1:-1])) * (-self.flowy[:,2:]) / np.maximum(self.water[:,1:], 0.001)
        boost_y2 = np.maximum(0, np.sign(self.flowy[:,1:-1])) * self.flowy[:,:-2] / np.maximum(self.water[:,:-1], 0.001)
        boost_y = np.maximum(boost_y1, boost_y2) / dx
        boost_y = np.minimum(1, boost_y)
        boost_y = np.exp(boost_y)
        self.flowy[:,1:-1] = self.flowy[:,1:-1] * (1 - frictionFactor * dt) + \
                             boost_y * ((self.water[:,:-1] + self.terrain[:,:-1]) - (self.water[:,1:] + self.terrain[:,1:])) * g * dt * dx

        # Overdraft mitigation
        total_outflow = np.maximum(0, -self.flowx[:-1]) + np.maximum(0, self.flowx[1:]) + \
            np.maximum(0, -self.flowy[:,:-1]) + np.maximum(0, self.flowy[:,1:])
        scale = (total_outflow == 0) + (total_outflow > 0) * (self.water * dx * dx / dt / np.maximum(total_outflow, 0.001))
        scale = np.minimum(1, scale)

        scaled_by_right = self.flowx[:-1] < 0
        scaled_by_left = self.flowx[1:] > 0
        scaled_xl = self.flowx[:-1] * scaled_by_right * scale
        scaled_xr = self.flowx[1:] * scaled_by_left * scale
        self.flowx[:-1] = scaled_xl
        self.flowx[-1] = 0
        self.flowx[1:] += scaled_xr

        scaled_by_bottom = self.flowy[:,:-1] < 0
        scaled_by_top = self.flowy[:,1:] > 0
        scaled_yb = self.flowy[:,:-1] * scaled_by_bottom * scale
        scaled_yt = self.flowy[:,1:] * scaled_by_top * scale
        self.flowy[:,:-1] = scaled_yb
        self.flowy[:,-1] = 0
        self.flowy[:,1:] += scaled_yt

        # Equivalent to this non-parallel version
        # for x, y in list(product(range(self.M), range(self.N))):
        #     total_outflow = 0  # m^3/s
        #     total_outflow += max(0, -self.flowx[x, y])
        #     total_outflow += max(0, -self.flowy[x, y])
        #     total_outflow += max(0, self.flowx[x + 1, y])
        #     total_outflow += max(0, self.flowy[x, y + 1])
        #
        #     if total_outflow > 0:
        #         max_outflow = self.water[x, y] * dx * dx / dt
        #         scale = min(1, max_outflow / total_outflow)
        #         if self.flowx[x, y] < 0:
        #             self.flowx[x, y] *= scale
        #         if self.flowx[x + 1, y] > 0:
        #             self.flowx[x + 1, y] *= scale
        #         if self.flowy[x, y] < 0:
        #             self.flowy[x, y] *= scale
        #         if self.flowy[x, y + 1] > 0:
        #             self.flowy[x, y + 1] *= scale

        # Move water
        self.water += (self.flowx[:-1] + self.flowy[:,:-1] - self.flowx[1:] - self.flowy[:,1:]) * dt / dx / dx

        # Outer border is cliff
        self.water[0,:] = 0
        self.water[-1,:] = 0
        self.water[:,0] = 0
        self.water[:,-1] = 0
