import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from PIL import Image

g = 10
width = 40
length = 40
dx = 1
v_damping = 0.99
baseline_dt = 0.1
dry_threshold = 1e-4
dry_threshold_disp = 1e-2

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
img = Image.open('C:/EarthSculptor/Maps/T.png')
img_resize = img.resize([M, N])
heights01 = np.array(img_resize, np.float32) / 65536
# center_dist = np.sqrt(np.square(Plot_Xs - width / 2) + np.square(Plot_Ys - length / 2))
# Z[1:-1,1:-1] = 10 * np.maximum(0, 0.66 - center_dist)
# Z = Z + np.transpose(Z)
Z[1:-1,1:-1] = heights01 * 10
H[34:37, 19:22] = 5

# Plotting setup
visual_interval = 10

def to_1d_index(x, y):
    return x + M * y

def to_2d_index(i):
    return i // M, i % M

tris = []
for i in range(M - 1):
    for j in range(N - 1):
        tris.append([to_1d_index(i, j), to_1d_index(i + 1, j), to_1d_index(i, j + 1)])
        tris.append([to_1d_index(i + 1, j), to_1d_index(i + 1, j + 1), to_1d_index(i, j + 1)])
masks = np.zeros([len(tris)])

fig = plt.figure()

ax_st = fig.add_subplot(2, 2, 1, projection='3d')
ax_st.set_title('Stage')
ax_h = fig.add_subplot(2, 2, 2)
ax_h.set_title('H')
ax_u = fig.add_subplot(2, 2, 3)
ax_u.set_title('U-abs')
ax_v = fig.add_subplot(2, 2, 4)
ax_v.set_title('V-abs')

np.set_printoptions(threshold=sys.maxsize, linewidth=np.nan)
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

# Simulation loop
def F(h, uv, z):
    return uv * uv / 2 + g * (h + z)

def fill_insufficient_segment_with_min(a: np.ndarray):
    from skimage.morphology import flood
    done = np.zeros_like(a)
    insufficient_map = a < 1
    done[np.where(insufficient_map == 0)] = 1
    while True:
        pending_x, pending_y = np.where(done == 0)
        if len(pending_x) == 0:
            return
        x0 = pending_x[0]
        y0 = pending_y[0]
        region = flood(insufficient_map, (x0, y0), connectivity=1)
        a[region] = np.min(a[region])
        done[region] = 1

def preserve_volume(H_Original, H_New, U, V, dt):
    dtdx = dt / dx

    Is_U_Upwind = (U[1:] + U[:-1]) > 0
    Upwind_H = Is_U_Upwind * H_Original[:-1,:] + (1 - Is_U_Upwind) * H_Original[1:,:]
    Flow_X = np.zeros([M + 1, N])
    Flow_X[1:-1, :] = (U[1:,:] + U[:-1,:]) / 2 * Upwind_H * dtdx

    Is_V_Upwind = (V[:,1:] + V[:,:-1]) > 0
    Upwind_H = Is_V_Upwind * H_Original[:,:-1] + (1 - Is_V_Upwind) * H_Original[:,1:]
    Flow_Y = np.zeros([M, N + 1])
    Flow_Y[:, 1:-1] = (V[:,:-1] + V[:,1:]) / 2 * Upwind_H * dtdx

    loop = 0
    while loop < 1000:
        Decrease = Flow_X[1:, :] - Flow_X[:-1, :] + Flow_Y[:, 1:] - Flow_Y[:, :-1]
        Proposed_H = H_Original - Decrease
        drying_ups = np.where(Proposed_H < -1e-6)  # M*N
        if loop == 0:
            print("N dry", len(drying_ups[0]))
        if len(drying_ups[0]) == 0:
            return Proposed_H, Flow_X, Flow_Y

        loop += 1
        cell = drying_ups[0][0], drying_ups[1][0]
        overdraft = -Proposed_H[cell]
        assert overdraft > 0

        cell_x, cell_y = cell

        flow_locations = []
        if Flow_X[cell_x, cell_y] < 0:
            flow_locations.append((Flow_X, cell_x, cell_y))
        if Flow_X[cell_x + 1, cell_y] > 0:
            flow_locations.append((Flow_X, cell_x + 1, cell_y))
        if Flow_Y[cell_x, cell_y] < 0:
            flow_locations.append((Flow_Y, cell_x, cell_y))
        if Flow_Y[cell_x, cell_y + 1] > 0:
            flow_locations.append((Flow_Y, cell_x, cell_y + 1))

        if flow_locations:
            outflow_sum = sum(abs(f[x,y]) for f,x,y in flow_locations)
            assert 1 - overdraft / outflow_sum > -1e-6
            for f, x, y in flow_locations:
                f[x,y] *= 1 - overdraft / outflow_sum
        else:
            print("Warning: No valid neighbors found for", cell_x, cell_y)
    print("Error: Loop exceeded limit")

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

    return H2, U2, V2

dt = baseline_dt
step = 0
H_Diff = 0

while True:
    if step % visual_interval == 0:
        title = '#%d Diff=%.3f V=%.2f MaxU=%.1f MaxV=%.1f dt=%.4f' % \
                (step, H_Diff, np.sum(H[1:-1, 1:-1]), np.max(np.abs(U)), np.max(np.abs(V)), dt)
        fig.suptitle(title)

        ax_st.clear()
        ax_st.set_zlim(0, 20)

        for index, (i1, i2, i3) in enumerate(tris):
            index_x1, index_y1 = to_2d_index(i1)
            index_x2, index_y2 = to_2d_index(i2)
            index_x3, index_y3 = to_2d_index(i3)
            dry = H[1 + index_x1, 1 + index_y1] < dry_threshold_disp \
                  and H[1 + index_x2, 1 + index_y2] < dry_threshold_disp \
                  and H[1 + index_x3, 1 + index_y3] < dry_threshold_disp
            masks[index] = dry

        water_triangulation = Triangulation(Plot_Xs.ravel(), Plot_Ys.ravel(), triangles=np.asarray(tris), mask=masks)
        stage = H[1:-1,1:-1] + Z[1:-1, 1:-1]
        ground_triangulation = Triangulation(Plot_Xs.ravel(), Plot_Ys.ravel(), triangles=np.asarray(tris), mask=1 - masks)
        ax_st.plot_trisurf(ground_triangulation, Z[1:-1, 1:-1].ravel(), color=(0.6, 0.4, 0, 0.4))
        ax_st.plot_trisurf(water_triangulation, stage.ravel(), color='b')

        ax_h.imshow(np.abs(H[1:-1]), cmap='hot', interpolation='nearest')
        ax_u.imshow(np.abs(U[1:-1]), cmap='hot', interpolation='nearest')
        ax_v.imshow(np.abs(V[1:-1]), cmap='hot', interpolation='nearest')

        plt.pause(0.001 if step > 0 else 3)

        # print('=====', title)
        # print('[H]')
        # print(H[1:-1, 1:-1])
        # print('[U]')
        # print(U[1:-1, 1:-1])
        # print('[V]')
        # print(V[1:-1, 1:-1])

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

    H2, U2, V2 = Lax_Wendroff(H, U, V, dt)

    if np.max(np.abs(U2)) * dt < dx / 2 and np.max(np.abs(V2)) * dt < dx / 2 \
            and dt * 2 < baseline_dt * 1.1:
        dt *= 2
        H2, U2, V2 = Lax_Wendroff(H, U, V, dt)
    while np.max(np.abs(U2)) < 1e3 and np.max(np.abs(V2)) < 1e3 \
            and (np.max(np.abs(U2)) * dt > dx or np.max(np.abs(V2)) * dt > dx):
        dt /= 2
        H2, U2, V2 = Lax_Wendroff(H, U, V, dt)
    if np.max(np.abs(U2)) >= 1e3 or np.max(np.abs(V2)) >= 1e3:
        raise RuntimeError("Speed blows up!!")

    H3, Flow_X, Flow_Y = preserve_volume(H[1:-1,1:-1], H2, U2, V2, dt)
    H3[np.where(H3 < dry_threshold)] = 0

    U2[np.where(H3 == 0)] = 0
    U_Bound = (Flow_X[1:, :] + Flow_X[:-1, :]) / 2 / np.maximum(H3, dry_threshold) * dx / dt
    U_OverBound = np.where(np.abs(U2) > np.abs(U_Bound))
    U2[U_OverBound] = U_Bound[U_OverBound]
    V2[np.where(H3 == 0)] = 0
    V_Bound = (Flow_Y[:, 1:] + Flow_Y[:, :-1]) / 2 / np.maximum(H3, dry_threshold) * dx / dt
    V_OverBound = np.where(np.abs(V2) > np.abs(V_Bound))
    V2[V_OverBound] = V_Bound[V_OverBound]

    # Apply change
    H_Diff = np.sum(np.abs(H3 - H[1:-1,1:-1]))
    H[1:-1, 1:-1] = H3
    U[1:-1, 1:-1] = U2
    V[1:-1, 1:-1] = V2

plt.show()
