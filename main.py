import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

import numpy as np
from tqdm import tqdm
from water import Simulator

simulation = Simulator(r'data\flat.txt')
STEP_SIZE = 100

paused = True
trigger = False  # single-step
def on_press(event):
    global paused, trigger
    if event.key == 'p':
        paused = not paused
    elif event.key == 'a':
        trigger = True

def on_close(event):
    exit(0)

fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', on_press)
fig.canvas.mpl_connect('close_event', on_close)
step = 0

colors = [
    (0.0, (0.6, 0.3, 0.0)),    # 0 -> brown
    (0.001, (0.7, 0.85, 1.0)), # 0.001 -> light blue
    (1.0, (0.0, 0.0, 0.5))     # 1.0 (normalized max) -> dark blue
]

custom_cmap = LinearSegmentedColormap.from_list('BrownToBlueSmooth', colors)

p = tqdm()
while True:
    ax.cla()

    surface = (simulation.water > 0.001) * (simulation.water + simulation.terrain)
    norm = Normalize(vmin=0, vmax=max(0.5, simulation.water.max()))
    ax.imshow(surface[1:-1,1:-1], cmap=custom_cmap, norm=norm, origin='upper',
              extent=(0.5,surface.shape[0]-1.5,
                      0.5,surface.shape[1]-1.5))

    max_v = max(np.max(np.abs(simulation.flowx)),
                np.max(np.abs(simulation.flowy)),
                0.001)
    for i in range(simulation.N):
        for j in range(1, simulation.N + 1):
            mag = simulation.flowy[i][j]
            mag = mag / max_v * 0.4
            ax.plot([j-0.5,j-0.5+mag],[i,i],color="red", linewidth=1)

    for i in range(1, simulation.N + 1):
        for j in range(simulation.N):
            mag = simulation.flowx[i][j]
            mag = mag / max_v * 0.4
            ax.plot([j, j], [i-0.5, i-0.5+mag], color="red", linewidth=1)

    ax.set_title(f"{step} maxF={max_v:.3f}")
    #plt.show()
    plt.pause(0.01)

    if not trigger and paused:
        plt.pause(1)
        continue

    step += STEP_SIZE
    trigger = False
    for _ in range(STEP_SIZE):
        p.update()
        simulation.step()
