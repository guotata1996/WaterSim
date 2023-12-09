import numpy as np
import sys
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize, linewidth=np.nan)
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

g = 10
width = 2
length = 2
dx = 0.1
v_damping = 0.99
baseline_dt = 0.01
dry_threshold = 0.01

M = int(width / dx)
N = int(length / dx)
# Initial condition
H = np.zeros([M + 2, N + 2])
U = np.zeros([M + 2, N + 2])
V = np.zeros([M + 2, N + 2])
Z = np.zeros([M + 2, N + 2])
H[:, :] = 1
H[:M//2, :N//2] = 2

def F(h, uv, z):
    return uv * uv / 2 + g * (h + z)

fig, ax1 = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
Plot_Xs = np.asarray(range(M)) * dx
Plot_Ys = np.asarray(range(N)) * dx
Plot_Xs, Plot_Ys = np.meshgrid(Plot_Xs, Plot_Xs)

dt = baseline_dt
step = 0
while True:
    step += 1

    # Boundary
    H[:, 0] = H[:, 1]
    H[:, N + 1] = H[:, N]
    H[0, :] = H[1, :]
    H[M + 1, :] = H[M, :]

    U *= v_damping
    U[:, 0] = U[:, 1]
    U[:, N + 1] = U[:, N]
    U[0, :] = -U[1, :]
    U[M + 1, :] = -U[M, :]

    V *= v_damping
    V[:, 0] = -V[:, 1]
    V[:, N + 1] = -V[:, N]
    V[0, :] = V[1, :]
    V[M + 1, :] = V[M, :]

    H_PlusZero = H[2:,1:-1]
    H_ZeroPlus = H[1:-1,2:]; H_ZeroZero = H[1:-1,1:-1]; H_ZeroMinus = H[1:-1,:-2]
    H_MinusZero = H[:-2,1:-1]

    U_PlusZero = U[2:,1:-1]
    U_ZeroPlus = U[1:-1,2:]; U_ZeroZero = U[1:-1,1:-1]; U_ZeroMinus = U[1:-1,:-2]
    U_MinusZero = U[:-2,1:-1]

    V_PlusZero = V[2:,1:-1]
    V_ZeroPlus = V[1:-1,2:]; V_ZeroZero = V[1:-1,1:-1]; V_ZeroMinus = V[1:-1,:-2]
    V_MinusZero = V[:-2,1:-1]

    Z_PlusZero = Z[2:,1:-1]
    Z_ZeroPlus = Z[1:-1,2:]; Z_ZeroZero = Z[1:-1,1:-1]; Z_ZeroMinus = Z[1:-1,:-2]
    Z_MinusZero = Z[:-2,1:-1]

    dtdx = dt / dx
    H1_PlusZero = (H_ZeroZero + H_PlusZero) / 2 \
                  - dtdx / 2 * (U_PlusZero * H_PlusZero - U_ZeroZero * H_ZeroZero)
    H1_ZeroMinus = (H_ZeroZero + H_ZeroMinus) / 2 \
                   - dtdx / 2 * (V_ZeroZero * H_ZeroZero - V_ZeroMinus * H_ZeroMinus)
    H1_ZeroPlus = (H_ZeroZero + H_ZeroPlus) / 2 \
                  - dtdx / 2 * (V_ZeroPlus * H_ZeroPlus - V_ZeroZero * H_ZeroZero)
    H1_MinusZero = (H_ZeroZero + H_MinusZero) / 2 \
                  - dtdx / 2 * (U_ZeroZero * H_ZeroZero - U_MinusZero * H_MinusZero)

    FU_ZeroZero = F(H_ZeroZero, U_ZeroZero, Z_ZeroZero)
    FV_ZeroZero = F(H_ZeroZero, V_ZeroZero, Z_ZeroZero)
    U1_PlusZero = (U_ZeroZero + U_PlusZero) / 2 \
                  - dtdx / 2 * (F(H_PlusZero, U_PlusZero, Z_PlusZero) - FU_ZeroZero)
    U1_ZeroMinus = (U_ZeroZero + U_ZeroMinus) / 2 \
                  - dtdx / 2 * V_ZeroZero * (U_ZeroZero - U_ZeroMinus)
    U1_ZeroPlus = (U_ZeroZero + U_ZeroPlus) / 2 \
                  - dtdx / 2 * V_ZeroZero * (U_ZeroPlus - U_ZeroZero)
    U1_MinusZero = (U_ZeroZero + U_MinusZero) / 2 \
                  - dtdx / 2 * (FU_ZeroZero - F(H_MinusZero, U_MinusZero, Z_MinusZero))

    V1_PlusZero = (V_ZeroZero + V_PlusZero) / 2 \
                  - dtdx / 2 * U_ZeroZero * (V_PlusZero - V_ZeroZero)
    V1_ZeroMinus = (V_ZeroZero + V_ZeroMinus) / 2 \
                  - dtdx / 2 * (FV_ZeroZero - F(H_ZeroMinus, V_ZeroMinus, Z_ZeroMinus))
    V1_ZeroPlus = (V_ZeroZero + V_ZeroPlus) / 2 \
                  - dtdx / 2 * (F(H_ZeroPlus, V_ZeroPlus, Z_ZeroPlus) - FV_ZeroZero)
    V1_MinusZero = (V_ZeroZero + V_MinusZero) / 2 \
                  - dtdx / 2 * U_ZeroZero * (V_ZeroZero - V_MinusZero)

    H2 = H_ZeroZero \
         - dtdx * (H1_PlusZero * U1_PlusZero - H1_MinusZero * U1_MinusZero) \
         - dtdx * (H1_ZeroPlus * V1_ZeroPlus - H1_ZeroMinus * V1_ZeroMinus)
    U2 = U_ZeroZero \
         - dtdx * (F(H1_PlusZero, U1_PlusZero, Z_PlusZero) - F(H1_MinusZero, U1_MinusZero, Z_MinusZero)) \
         - dtdx * (V1_PlusZero + V1_ZeroMinus + V1_ZeroPlus + V1_MinusZero) / 4 * (U1_ZeroPlus - U1_ZeroMinus)
    V2 = V_ZeroZero \
         - dtdx * (U1_PlusZero + U1_ZeroMinus + U1_ZeroPlus + U1_MinusZero) / 4 * (V1_PlusZero - V1_MinusZero) \
         - dtdx * (F(H1_ZeroPlus, V1_ZeroPlus, Z_ZeroPlus) - F(H1_ZeroMinus, V1_ZeroMinus, Z_ZeroMinus))

    # Apply change
    H_Diff = np.sum(np.abs(H2 - H[1:-1,1:-1]))
    H[1:-1, 1:-1] = H2
    U[1:-1, 1:-1] = U2
    V[1:-1, 1:-1] = V2

    title = '#%d Diff=%.3f V=%.2f MaxU=%.1f dt=%.4f' % (step, H_Diff, np.sum(H[1:-1,1:-1]), np.max(np.abs(U)), dt)
    fig.suptitle(title)

    ax1.clear()
    ax1.set_zlim(0, 5)
    ax1.plot_surface(Plot_Xs, Plot_Ys, H[1:-1,1:-1], linewidth=0, antialiased=False)

    plt.pause(0.001)

    # print('=====', title)
    print('[H]')
    print(H[1:-1,1:-1])
    # print('[U]')
    # print(U[1:-1,1:-1])
    # print('[V]')
    # print(V[1:-1,1:-1])
    #
    # input('continue...')

plt.show()