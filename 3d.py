import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

g = 10
width = 2
length = 2
dx = 0.1
v_damping = 0.99
baseline_dt = 0.005
dry_threshold = 0.01

M = int(width / dx)
N = int(length / dx)
H = np.zeros([M + 2, N + 2])
U = np.zeros([M + 2, N + 2])
V = np.zeros([M + 2, N + 2])
Z = np.zeros([M + 2, N + 2])
Plot_Xs = np.asarray(range(M)) * dx
Plot_Ys = np.asarray(range(N)) * dx
Plot_Xs, Plot_Ys = np.meshgrid(Plot_Xs, Plot_Xs)

# Initial condition
center_dist = np.sqrt(np.square(Plot_Xs - width / 2) + np.square(Plot_Ys - length / 2))
Z[1:-1,1:-1] = 10 * np.maximum(0, 0.66 - center_dist)
Z = Z + np.transpose(Z)
H[:M//2, :N//2] = 10

# Plotting setup
visual_interval = 20

def to_1d_index(x, y):
    return x + M * y

def to_2d_index(i):
    return i % M, i // M

tris = []
for i in range(M - 1):
    for j in range(N - 1):
        tris.append([to_1d_index(i, j), to_1d_index(i + 1, j), to_1d_index(i, j + 1)])
        tris.append([to_1d_index(i + 1, j), to_1d_index(i + 1, j + 1), to_1d_index(i, j + 1)])
masks = np.zeros([len(tris)])

fig, ax1 = plt.subplots(1, 1, subplot_kw={"projection": "3d"})

np.set_printoptions(threshold=sys.maxsize, linewidth=np.nan)
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

# Simulation loop
def F(h, uv, z):
    return uv * uv / 2 + g * (h + z)

def Lax_Wendroff(H, U, V, dt):
    H_PlusZero = H[2:, 1:-1]
    H_ZeroPlus = H[1:-1, 2:]
    H_ZeroZero = H[1:-1, 1:-1]
    H_ZeroMinus = H[1:-1, :-2]
    H_MinusZero = H[:-2, 1:-1]

    U_PlusZero = U[2:, 1:-1]
    U_ZeroPlus = U[1:-1, 2:]
    U_ZeroZero = U[1:-1, 1:-1]
    U_ZeroMinus = U[1:-1, :-2]
    U_MinusZero = U[:-2, 1:-1]

    V_PlusZero = V[2:, 1:-1]
    V_ZeroPlus = V[1:-1, 2:]
    V_ZeroZero = V[1:-1, 1:-1]
    V_ZeroMinus = V[1:-1, :-2]
    V_MinusZero = V[:-2, 1:-1]

    Z_PlusZero = Z[2:, 1:-1]
    Z_ZeroPlus = Z[1:-1, 2:]
    Z_ZeroZero = Z[1:-1, 1:-1]
    Z_ZeroMinus = Z[1:-1, :-2]
    Z_MinusZero = Z[:-2, 1:-1]

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

    # Preserve Volume (prevent negative H)
    Flow_Inner_X = U2 * H2 * dtdx
    Flow_Inner_X = (Flow_Inner_X[1:, :] + Flow_Inner_X[:-1, :]) / 2
    Flow_X = np.zeros([M + 1, N])
    Flow_X[1:-1, :] = Flow_Inner_X
    Flow_Inner_Y = V2 * H2 * dtdx
    Flow_Inner_Y = (Flow_Inner_Y[:, 1:] + Flow_Inner_Y[:, :-1]) / 2
    Flow_Y = np.zeros([M, N + 1])
    Flow_Y[:, 1:-1] = Flow_Inner_Y
    Decrease = Flow_X[1:, :] - Flow_X[:-1, :] + Flow_Y[:, 1:] - Flow_Y[:, :-1]
    Decrease_Mul = np.minimum(H_ZeroZero / np.maximum(Decrease, 1e-5), 1)
    Decrease_Mul[np.where(Decrease <= 0)] = 1

    Flow_Mul_X1 = np.zeros([M + 1, N])
    Flow_Mul_X2 = np.zeros([M + 1, N])
    Flow_Mul_X1[1:, :] = Decrease_Mul
    Flow_Mul_X2[:-1, :] = Decrease_Mul
    Flow_Mul_X = np.minimum(Flow_Mul_X1, Flow_Mul_X2)
    Flow_X *= Flow_Mul_X
    Flow_Mul_Y1 = np.zeros([M, N + 1])
    Flow_Mul_Y2 = np.zeros([M, N + 1])
    Flow_Mul_Y1[:, 1:] = Decrease_Mul
    Flow_Mul_Y2[:, :-1] = Decrease_Mul
    Flow_Mul_Y = np.minimum(Flow_Mul_Y1, Flow_Mul_Y2)
    Flow_Y *= Flow_Mul_Y

    Decrease1 = Flow_X[1:, :] - Flow_X[:-1, :] + Flow_Y[:, 1:] - Flow_Y[:, :-1]
    H3 = H2 - Decrease1
    H3[np.where(H3 < dry_threshold)] = 0

    U_Bound = (Flow_X[1:, :] + Flow_X[:-1, :]) / 2 / np.maximum(H3, dry_threshold) * dx / dt
    U_OverBound = np.where(np.abs(U2) > np.abs(U_Bound))
    U2[U_OverBound] = U_Bound[U_OverBound]
    V_Bound = (Flow_Y[:, 1:] + Flow_Y[:, :-1]) / 2 / np.maximum(H3, dry_threshold) * dx / dt
    V_OverBound = np.where(np.abs(V2) > np.abs(V_Bound))
    V2[V_OverBound] = V_Bound[V_OverBound]

    return H3, U2, V2

dt = baseline_dt
step = 0
H_Diff = 0

while True:
    if step % visual_interval == 0:
        title = '#%d Diff=%.3f V=%.2f MaxU=%.1f dt=%.4f' % (step, H_Diff, np.sum(H[1:-1, 1:-1]), np.max(np.abs(U)), dt)
        fig.suptitle(title)

        ax1.clear()
        ax1.set_zlim(0, 20)

        for index, (i1, i2, i3) in enumerate(tris):
            index_x1, index_y1 = to_2d_index(i1)
            index_x2, index_y2 = to_2d_index(i2)
            index_x3, index_y3 = to_2d_index(i3)
            dry = H[1 + index_x1, 1 + index_y1] == 0 and H[1 + index_x2, 1 + index_y2] == 0 and H[1 + index_x3, 1 + index_y3] == 0
            masks[index] = dry

        water_triangulation = Triangulation(Plot_Xs.ravel(), Plot_Ys.ravel(), triangles=np.asarray(tris), mask=masks)
        stage = H[1:-1,1:-1] + Z[1:-1, 1:-1]
        ground_triangulation = Triangulation(Plot_Xs.ravel(), Plot_Ys.ravel(), triangles=np.asarray(tris), mask=1 - masks)
        ax1.plot_trisurf(ground_triangulation, Z[1:-1, 1:-1].ravel(), color=(0.6,0.4,0,0.4))
        ax1.plot_trisurf(water_triangulation, stage.ravel(), color='b')

        plt.pause(0.001)

        print('=====', title)
        print('[H]')
        print(H[1:-1, 1:-1])
        print('[U]')
        print(U[1:-1, 1:-1])
        print('[V]')
        print(V[1:-1, 1:-1])
        #
        # input('continue...')

    step += 1

    # Boundary
    H[1:-1, 0] = H[1:-1, 1]
    H[1:-1, N + 1] = H[1:-1, N]
    H[0, 1:-1] = H[1, 1:-1]
    H[M + 1, 1:-1] = H[M, 1:-1]

    U *= v_damping
    U[:, 0] = -U[:, 1]
    U[:, N + 1] = -U[:, N]
    U[0, :] = U[1, :]
    U[M + 1, :] = U[M, :]

    V *= v_damping
    V[:, 0] = V[:, 1]
    V[:, N + 1] = V[:, N]
    V[0, :] = -V[1, :]
    V[M + 1, :] = -V[M, :]

    H_New, U_New, V_New = Lax_Wendroff(H, U, V, dt)

    if np.max(np.abs(U_New)) * dt < dx / 2 and np.max(np.abs(V_New)) * dt < dx / 2 \
            and dt * 2 < baseline_dt * 1.1:
        dt *= 2
        H_New, U_New, V_New = Lax_Wendroff(H, U, V, dt)
    while np.max(np.abs(U_New)) < 1e3 and np.max(np.abs(V_New)) < 1e3 \
            and (np.max(np.abs(U_New)) * dt > dx or np.max(np.abs(V_New)) * dt > dx):
        dt /= 2
        H_New, U_New, V_New = Lax_Wendroff(H, U, V, dt)
    if np.max(np.abs(U_New)) >= 1e3 or np.max(np.abs(V_New)) >= 1e3:
        raise RuntimeError("Speed blows up!!")

    # Apply change
    H_Diff = np.sum(np.abs(H_New - H[1:-1,1:-1]))
    H[1:-1, 1:-1] = H_New
    U[1:-1, 1:-1] = U_New
    V[1:-1, 1:-1] = V_New

plt.show()
