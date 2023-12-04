import numpy as np
import matplotlib.pyplot as plt

g = 10
N = 20
dt = 0.01
dx = 0.1
dtdx = dt / dx
H = np.empty([N + 2])
H_N1 = np.empty([N + 2])
U = np.zeros([N + 2])

# Initial condition
Z = 0.5 - np.asarray(np.linspace(0, 0.5, N + 2))
H[N // 3: N // 3 * 2] = 0.5

# Reporting
x_axis = np.asarray(range(N)) * dx
fig, (ax1, ax2) = plt.subplots(2, 1)

H_Diff = 0
report_interval = 10

for step in range(0, 1000):
    # Boundary
    H[0] = H[1]
    H[N + 1] = H[N]
    U[0] = -U[1]
    U[N + 1] = -U[N]

    im_depth = 0.01
    H_NonDry = H + im_depth

    # Eq 1
    H_N1[1: -1] = H_NonDry[1: -1] - dtdx / 2 \
                  * (H_NonDry[1: -1] * (U[2:] - U[: -2]) + U[1: -1] * (H_NonDry[2:] - H_NonDry[: -2]))

    # Eq 2
    U[1: -1] = \
        (U[: -2] + U[2:]) / 2 \
        - U[1: -1] / H_NonDry[1: -1] * (H_N1[1: -1] - H_NonDry[1: -1]) \
        - U[1: -1] * dtdx * (U[2:] - U[: -2]) \
        - dtdx / 2 * (U[1: -1] * U[1: -1] / H_NonDry[1: -1] + g) * (H_NonDry[2:] - H_NonDry[: -2]) \
        - dtdx / 2 * g * (Z[2:] - Z[: -2])

    H_Dry = np.maximum(H_N1[1: -1] - im_depth, 0)
    H_Diff += np.sum(np.abs(H_Dry - H[1:-1]))
    H[1: -1] = H_Dry
    U[np.where(H == 0)] = 0

    if step % report_interval == 0:
        H_DiffAvg = H_Diff / report_interval
        H_Diff = 0
        title = '#%d Diff=%.3f V=%.2f MaxU=%.1f' % (step, H_DiffAvg, sum(H[1: -1]), np.max(U))
        fig.suptitle(title)

        ax1.clear()
        ax1.set_ylim(0, 1)
        ax1.bar(x_axis, Z[1: -1] + H[1: -1], width=dx, color=(0, 0.4, 1, 0.4))
        ax1.bar(x_axis, Z[1: -1], width=dx, color=(0.6, 0.4, 0, 1))

        ax2.clear()
        ax2.set_ylim(-1, 1)
        ax2.scatter(x_axis, U[1: -1], color=(0.6, 0.4, 0.1))

        plt.pause(0.001)

        print('=====', title)
        print('[H]', H[1: -1])
        print('[U]', U[1: -1])
        # input('continue...')

plt.show()
