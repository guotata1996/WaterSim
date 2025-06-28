import numpy as np
from itertools import product
from vxl_importer import import_vxl

frictionFactor = 0.1
g = 0.1
dx = 1
dt_mult:int = 5
sourceRate = 0.2 # m/s

class Simulator:
    BASE_DT = 0.1

    def __init__(self, vxl:str):
        terrain, water, source = import_vxl(vxl)
        self.M = terrain.shape[0] + 2
        self.N = terrain.shape[1] + 2

        self.terrain = np.zeros((self.M,self.N))
        self.terrain[1:-1,1:-1] = terrain

        self.water = np.zeros((self.M,self.N))  # m
        self.water[1:-1,1:-1] = water

        self.source = {}
        for (x, y), intensity in source.items():
            self.source[(x+1,y+1)] = intensity

        self.flowx = np.zeros([self.M + 1, self.N]) #m^3/s
        self.flowy = np.zeros([self.M, self.N + 1])
        self.dt = dt_mult * Simulator.BASE_DT
        self.time = 0

    def step(self):
        # Source injection
        for (x, y), intensity in self.source.items():
            self.water[x,y] += intensity * self.dt * sourceRate

        # Flow change due to gravity
        # boost causes varying result when scaling in length
        boost_x1 = np.maximum(0, -np.sign(self.flowx[1:-1,:])) * (-self.flowx[2:,:]) / np.maximum(self.water[1:,:], 0.001) # m^2/s
        boost_x2 = np.maximum(0, np.sign(self.flowx[1:-1,:])) * self.flowx[:-2,:] / np.maximum(self.water[:-1,:], 0.001)
        boost_x = np.maximum(boost_x1, boost_x2) / dx
        boost_x = np.minimum(1, boost_x)
        boost_x = np.exp(boost_x)

        # a(a(af+b)+b)+b = a^3 f + (a^2+a+1)b = a^3 f + coeff_const * b
        coeff_const = (1 - (1 - frictionFactor) ** dt_mult) / frictionFactor
        self.flowx[1:-1] = self.flowx[1:-1] * ((1 - frictionFactor) ** dt_mult) + coeff_const * \
                           boost_x * ((self.water[:-1, :] + self.terrain[:-1:]) - (self.water[1:, :] + self.terrain[1:, :])) * g * dx
        # Equivalent to this iterative version
        # for _ in range(dt_mult):
        #     self.flowx[1:-1] = self.flowx[1:-1] * (1 - frictionFactor) + \
        #                        boost_x * ((self.water[:-1,:] + self.terrain[:-1:]) - (self.water[1:,:] + self.terrain[1:,:])) * g * dx

        boost_y1 = np.maximum(0, -np.sign(self.flowy[:,1:-1])) * (-self.flowy[:,2:]) / np.maximum(self.water[:,1:], 0.001)
        boost_y2 = np.maximum(0, np.sign(self.flowy[:,1:-1])) * self.flowy[:,:-2] / np.maximum(self.water[:,:-1], 0.001)
        boost_y = np.maximum(boost_y1, boost_y2) / dx
        boost_y = np.minimum(1, boost_y)
        boost_y = np.exp(boost_y)

        self.flowy[:,1:-1] = self.flowy[:,1:-1] * ((1 - frictionFactor) ** dt_mult) + coeff_const * \
                           boost_y * ((self.water[:,:-1] + self.terrain[:,:-1]) - (self.water[:,1:] + self.terrain[:,1:])) * g * dx
        # Equivalent to this iterative version
        # for _ in range(dt_mult):
        #     self.flowy[:,1:-1] = self.flowy[:,1:-1] * (1 - frictionFactor) + \
        #                          boost_y * ((self.water[:,:-1] + self.terrain[:,:-1]) - (self.water[:,1:] + self.terrain[:,1:])) * g * dx

        # Overdraft mitigation
        total_outflow = np.maximum(0, -self.flowx[:-1]) + np.maximum(0, self.flowx[1:]) + \
            np.maximum(0, -self.flowy[:,:-1]) + np.maximum(0, self.flowy[:,1:])
        scale = (total_outflow == 0) + (total_outflow > 0) * (self.water * dx * dx / self.dt / np.maximum(total_outflow, 0.001))
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
        self.water += (self.flowx[:-1] + self.flowy[:,:-1] - self.flowx[1:] - self.flowy[:,1:]) * self.dt / dx / dx

        # Outer border is cliff
        self.water[0,:] = 0
        self.water[-1,:] = 0
        self.water[:,0] = 0
        self.water[:,-1] = 0

        self.time += self.dt
