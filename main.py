import matplotlib.pyplot as plt
import numpy as np
import tqdm

from vxl_importer import import_vxl

terrain, water = import_vxl(r'data\lake_fill.txt')
N = terrain.shape[0]
flowx = np.zeros([N + 1, N])
flowy = np.zeros([N, N + 1])
frictionFactor = 0.5
g = 10
dx = dy = 10
dt = 1

paused = False
def on_press(event):
    global paused
    if event.key == 'p':
        paused = not paused

def on_close(event):
    exit(0)

fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', on_press)
fig.canvas.mpl_connect('close_event', on_close)

for step in tqdm.tqdm(range(100000)):
    if paused:
        plt.pause(1)
        continue

    flowx[0, :] = 0
    flowx[N, :] = 0
    flowy[:, 0] = 0
    flowy[:, N] = 0

    for y in range(N):
        for x in range(1, N):
            flowx[x, y] = flowx[x, y] * frictionFactor + \
                          ((water[x - 1, y] + terrain[x - 1, y]) - (water[x, y] + terrain[x, y])) * g * dt / dx

    for y in range(1, N):
        for x in range(N):
            flowy[x, y] = flowy[x, y] * frictionFactor + \
                          ((water[x, y - 1] + terrain[x, y - 1]) - (water[x, y] + terrain[x, y])) * g * dt / dx

    for y in range(N):
        for x in range(N):
            total_outflow = 0
            total_outflow += max(0, -flowx[x,y])
            total_outflow += max(0, -flowy[x,y])
            total_outflow += max(0, flowx[x+1,y])
            total_outflow += max(0, flowy[x,y+1])

            if total_outflow > 0:
                max_outflow = water[x, y] * dx * dy / dt
                scale = min(1, max_outflow / total_outflow)
                if flowx[x, y] < 0:
                    flowx[x, y] *= scale
                if flowx[x + 1, y] > 0:
                    flowx[x + 1, y] *= scale
                if flowy[x, y] < 0:
                    flowy[x, y] *= scale
                if flowy[x, y + 1] > 0:
                    flowy[x, y + 1] *= scale

    for y in range(N):
        for x in range(N):
            water[x, y] += (flowx[x, y] + flowy[x, y] - flowx[x+1, y] - flowy[x, y+1]) * dt/dx/dy

    if step % 500 == 0:
        ax.imshow(water, cmap='hot', interpolation='nearest')
        ax.set_title(str(step))
        #plt.show()
        plt.pause(0.01)
