import numpy as np
import matplotlib.pyplot as plt

g = 10
N = 20
dt = 0.01
dx = 0.1
dtdx = dt / dx
H = np.empty([N + 2])
U = np.zeros([N + 2])
v_damping = 0.99
dry_threshold = 0.01

# Initial condition
Z = 5 - np.asarray(np.linspace(0, 5, N + 2))
# Z = np.zeros([N + 2])
H[N // 3: N // 3 * 2] = 5

# Reporting
x_axis = np.asarray(range(N)) * dx
fig, (ax1, ax2) = plt.subplots(2, 1)

H_Diff = 0
report_interval = 10

def F_U(u, h, z):
    return u * u / 2 + g * (h + z)

for step in range(0, 2000):
    # Boundary
    H[0] = H[1]
    H[N + 1] = H[N]
    U *= v_damping
    U[0] = -U[1]
    U[N + 1] = -U[N]

    Hj = H[1: -1]
    H_Plus = H[2:]
    H_Minus = H[:-2]
    Uj = U[1: -1]
    U_Plus = U[2:]
    U_Minus = U[:-2]
    Zj = Z[1: -1]
    Z_Plus = Z[2:]
    Z_Minus = Z[:-2]

    # Lax-Wendroff two step
    H_Plus2 = (Hj + H_Plus) / 2 - dtdx / 2 * (H_Plus * U_Plus - Hj * Uj)
    H_Minus2 = (H_Minus + Hj) / 2 - dtdx / 2 * (Hj * Uj - H_Minus * U_Minus)
    U_Plus2 = (Uj + U_Plus) / 2 - dtdx / 2 * (F_U(U_Plus, H_Plus, Z_Plus) - F_U(Uj, Hj, Zj))
    U_Minus2 = (U_Minus + Uj) / 2 - dtdx / 2 * (F_U(Uj, Hj, Zj) - F_U(U_Minus, H_Minus, Z_Minus))
    Z_Plus2 = (Zj + Z_Plus) / 2
    Z_Minus2 = (Z_Minus + Zj) / 2

    H_New = Hj - dtdx * (H_Plus2 * U_Plus2 - H_Minus2 * U_Minus2)
    U_New = Uj - dtdx * (F_U(U_Plus2, H_Plus2, Z_Plus2) - F_U(U_Minus2, H_Minus2, Z_Minus2))
    Flow = U_New * H_New * dtdx
    Flow = (Flow[1:] + Flow[:-1]) / 2
    Flow = np.concatenate([[0], Flow, [0]])

    # Preserve Volume (prevent negative H)
    Decrease = Flow[1:] - Flow[:-1]
    Decrease_Mul = np.minimum(H[1: -1] / np.maximum(Decrease, 1e-3), 1)
    Decrease_Mul[np.where(Decrease <= 0)] = 1
    Flow_Mul = np.minimum(np.concatenate([[0], Decrease_Mul]), np.concatenate([Decrease_Mul, [0]]))
    Flow = Flow * Flow_Mul
    Decrease_Adj = Flow[1:] - Flow[:-1]

    H_N1 = H[1:-1] - Decrease_Adj
    H_N1 = np.maximum(H_N1, 0)
    H_N1[np.where(H_N1 < dry_threshold)] = 0

    # Apply change
    H_Diff += np.sum(np.abs(H_N1 - H[1:-1]))
    H[1:-1] = H_N1
    U[1:-1] = U_New
    U[np.where(H == 0)] = 0

    if step % report_interval == 0:
        H_DiffAvg = H_Diff / report_interval
        H_Diff = 0
        title = '#%d Diff=%.3f V=%.2f MaxU=%.1f' % (step, H_DiffAvg, sum(H[1: -1]), np.max(U))
        fig.suptitle(title)

        ax1.clear()
        ax1.set_ylim(0, 5)
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
