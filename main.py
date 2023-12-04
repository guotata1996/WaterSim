import numpy as np
import matplotlib.pyplot as plt

g = 10
N = 20

dt = 0.01
dx = 0.1
H = np.empty([N + 2])
H_N1 = np.empty([N + 2])
U = np.zeros([N + 2])
Z = np.linspace(0, 0.3, N + 2)
H[1: N // 3] = 0.5
H[N // 3:] = 0.5

x_axis = np.asarray(range(N)) * dx

dtdx = dt / dx

fig, (ax1, ax2) = plt.subplots(2, 1)

for step in range(0, 1000):
    # Boundary
    H[0] = H[1]
    H[N + 1] = H[N]
    U[0] = -U[1]
    U[N + 1] = -U[N]

    # Eq 1
    H_N1[1: -1] = H[1: -1] \
                  - dtdx / 2 * (H[1: -1] * (U[2:] - U[: -2]) + U[1: -1] * (H[2:] - H[: -2]))

    # Eq 2
    U[1: -1] = \
        (U[: -2] + U[2:]) / 2 \
        - U[1: -1] / H[1: -1] * (H_N1[1: -1] - H[1: -1]) \
        - U[1: -1] * dtdx * (U[2:] - U[: -2]) \
        - dtdx / 2 * (U[1: -1] * U[1: -1] / H[1: -1] + g) * (H[2:] - H[: -2]) \
        - dtdx / 2 * g * (Z[2:] - Z[: -2])

    H[1: -1] = H_N1[1: -1]

    if step % 10 == 0:
        fig.suptitle('step {0} volume {1}'.format(step, sum(H[1: -1])))

        ax1.clear()
        ax1.set_ylim(0, 1)
        ax1.bar(x_axis, Z[1: -1] + H[1: -1], width=dx, color=(0, 0.4, 1, 0.4))
        ax1.bar(x_axis, Z[1: -1], width=dx, color=(0.6, 0.4, 0, 1))

        ax2.clear()
        ax2.set_ylim(-1, 1)
        ax2.scatter(x_axis, U[1: -1], color=(0.6, 0.4, 0.1))

        plt.pause(0.001)

        # print('==========STEP [', step, '] Sum=', sum(H[1: -1]), "==========")
        # print('[H]', H[1: -1])
        # print('[U]', U[1: -1])
        # input('continue...')

plt.show()
