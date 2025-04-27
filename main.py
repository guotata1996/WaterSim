import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from water import Simulator

simulation = Simulator(r'data\channel.txt')
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

p = tqdm()
while True:
    ax.cla()
    ax.imshow(simulation.water, cmap='hot', interpolation='nearest')

    max_v = max(np.max(np.abs(simulation.flowx)),
                np.max(np.abs(simulation.flowy)),
                0.001)
    for i in range(simulation.N):
        for j in range(1, simulation.N + 1):
            mag = simulation.flowy[i][j]
            mag = mag / max_v * 0.4
            ax.plot([j-0.5,j-0.5+mag],[i,i],color="cyan", linewidth=0.5)

    for i in range(1, simulation.N + 1):
        for j in range(simulation.N):
            mag = simulation.flowx[i][j]
            mag = mag / max_v * 0.4
            ax.plot([j, j], [i-0.5, i-0.5+mag], color="cyan", linewidth=0.5)

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
