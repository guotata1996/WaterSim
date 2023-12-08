import numpy as np
import matplotlib.pyplot as plt

g = 10
N = 50
baseline_dt = 0.0005  # For large max depth, use smaller value
dx = 0.1
v_damping = 0.99
dry_threshold = 0.01

# Initial condition
H = np.zeros([N + 2])
U = np.zeros([N + 2])
Z = np.zeros([N + 2])
# Z[N // 4: N // 2] = np.linspace(0, 15, len(Z[N // 4: N // 2]))
# Z[N // 2: N * 3 // 4] = np.linspace(15, 0, len(Z[N // 2: N * 3 // 4]))
H[:N // 4] = 40

# Reporting
report_interval = 100
manual_stepping = False

H_Diff = 0
x_axis = np.asarray(range(N)) * dx
fig, (ax1, ax2) = plt.subplots(2, 1)

def F_U(u, h, z):
    return u * u / 2 + g * (h + z)

def Lax_Wendroff(H, U, Z, dt):
    Hj = H[1: -1]
    H_Plus = H[2:]
    H_Minus = H[:-2]
    Uj = U[1: -1]
    U_Plus = U[2:]
    U_Minus = U[:-2]
    Zj = Z[1: -1]
    Z_Plus = Z[2:]
    Z_Minus = Z[:-2]

    dtdx = dt / dx
    H_Plus2 = (Hj + H_Plus) / 2 - dtdx / 2 * (H_Plus * U_Plus - Hj * Uj)
    H_Minus2 = (H_Minus + Hj) / 2 - dtdx / 2 * (Hj * Uj - H_Minus * U_Minus)
    U_Plus2 = (Uj + U_Plus) / 2 - dtdx / 2 * (F_U(U_Plus, H_Plus, Z_Plus) - F_U(Uj, Hj, Zj))
    U_Minus2 = (U_Minus + Uj) / 2 - dtdx / 2 * (F_U(Uj, Hj, Zj) - F_U(U_Minus, H_Minus, Z_Minus))
    Z_Plus2 = (Zj + Z_Plus) / 2
    Z_Minus2 = (Z_Minus + Zj) / 2

    H_New = Hj - dtdx * (H_Plus2 * U_Plus2 - H_Minus2 * U_Minus2)
    U_New = Uj - dtdx * (F_U(U_Plus2, H_Plus2, Z_Plus2) - F_U(U_Minus2, H_Minus2, Z_Minus2))
    return np.concatenate([[H[0]], H_New, [H[-1]]]), np.concatenate([[U[0]], U_New, [U[-1]]])

def FTCS(H, U, Z, dt):
    dtdx = dt / dx
    H_New = H[1:-1] - dtdx / 2 * (U[2:] * H[2:] - U[:-2] * H[:-2])
    U_New = U[1:-1] - dtdx / 2 * (F_U(U[2:], H[2:], Z[2:]) - F_U(U[:-2], H[:-2], Z[:-2]))
    H_New[np.where(H_New < dry_threshold)] = 0
    return np.concatenate([[H[0]], H_New, [H[-1]]]), np.concatenate([[U[0]], U_New, [U[-1]]])

step_function = FTCS

dt = baseline_dt
for step in range(0, 100000):
    # H[1] += 100 * dt
    # H[N] = 0

    # Boundary
    H[0] = H[1]
    H[N + 1] = H[N]
    U *= v_damping
    U[0] = -U[1]
    U[N + 1] = -U[N]

    H_New, U_New = step_function(H, U, Z, dt)

    if np.max(np.abs(U_New)) * dt < dx / 2 and dt * 2 < baseline_dt * 1.1:
        dt *= 2
        H_New, U_New = step_function(H, U, Z, dt)
    while np.max(np.abs(U_New)) < 1e3 and np.max(np.abs(U_New)) * dt > dx:
        dt /= 2
        H_New, U_New = step_function(H, U, Z, dt)
    if np.max(np.abs(U_New)) >= 1e3:
        raise RuntimeError("Speed blows up!!")

    # Preserve Volume (prevent negative H)
    if step_function == Lax_Wendroff:
        U_New = U_New[1:-1]
        H_New = H_New[1:-1]
        Flow = U_New * H_New * dt / dx
        Flow = (Flow[1:] + Flow[:-1]) / 2
        Flow = np.concatenate([[0], Flow, [0]])

        Decrease = Flow[1:] - Flow[:-1]
        Decrease_Mul = np.minimum(H[1: -1] / np.maximum(Decrease, 1e-3), 1)
        Decrease_Mul[np.where(Decrease <= 0)] = 1
        Flow_Mul = np.minimum(np.concatenate([[0], Decrease_Mul]), np.concatenate([Decrease_Mul, [0]]))
        Flow = Flow * Flow_Mul
        Decrease_Adj = Flow[1:] - Flow[:-1]

        H_New = H[1:-1] - Decrease_Adj
        H_New = np.maximum(H_New, 0)
        H_New[np.where(H_New < dry_threshold)] = 0

        # Eliminate bouncing U near shoreline
        U_Bound = (Flow[1:] + Flow[:-1]) / 2 / np.maximum(H_New, dry_threshold) * dx / dt
        U_New[np.where(np.abs(U_New) > np.abs(U_Bound))] = U_Bound[np.where(np.abs(U_New) > np.abs(U_Bound))]

        H_New = np.concatenate([[H[0]], H_New, [H[-1]]])
        U_New = np.concatenate([[U[0]], U_New, [U[-1]]])

    # Apply change
    H_Diff += np.sum(np.abs(H_New - H))
    H = H_New
    U = U_New
    U[np.where(H == 0)] = 0

    if step % report_interval == 0:
        H_DiffAvg = H_Diff / report_interval
        H_Diff = 0
        title = '#%d Diff=%.3f V=%.2f MaxU=%.1f dt=%.4f' % (step, H_DiffAvg, sum(H[1: -1]), np.max(np.abs(U)), dt)
        fig.suptitle(title)

        ax1.clear()
        ax1.set_ylim(0, 40)
        ax1.bar(x_axis, Z[1: -1] + H[1: -1], width=dx, color=(0, 0.4, 1, 0.4))
        ax1.bar(x_axis, Z[1: -1], width=dx, color=(0.6, 0.4, 0, 1))

        ax2.clear()
        ax2.set_ylim(-5, 5)
        ax2.scatter(x_axis, U[1: -1], color=(0.6, 0.4, 0.1))

        plt.pause(0.001)

        print('=====', title)
        print('[H]', H[1: -1])
        print('[U]', U[1: -1])
        if manual_stepping:
            input('continue...')

plt.show()
