import matplotlib.pyplot as plt
from water import Simulator

simulation = Simulator(r'data\lake_fill.txt')

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

while True:
    ax.imshow(simulation.water, cmap='hot', interpolation='nearest')
    ax.set_title(str(step))
    #plt.show()
    plt.pause(0.01)

    if not trigger and paused:
        plt.pause(1)
        continue

    step += 100
    trigger = False
    for _ in range(100):
        simulation.step()
