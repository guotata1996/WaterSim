import numpy as np
import sys
import open3d as o3d
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
Z = np.ones([M + 2, N + 2]) * 50

Plot_Xs = []
Plot_Ys = []
grid_centers = []

# Initial condition
img = Image.open('C:/EarthSculptor/Maps/T.png')
img_resize = img.resize([M, N])
heights01 = np.array(img_resize, np.float32) / 65536
Z[1:-1,1:-1] = heights01 * 10
H[34:37, 19:22] = 15

# Plotting setup
visual_interval = 10

def to_1d_index(x, y):
    return x + M * y

def to_2d_index(i):
    return i // M, i % M

tris = []
for j in range(N):
    for i in range(M):
        grid_centers.append([(i + 0.5) * dx, (j + 0.5) * dx, 0])
        base_v = len(Plot_Xs)
        Plot_Xs.append(i * dx); Plot_Ys.append(j * dx)
        Plot_Xs.append((i + 1) * dx); Plot_Ys.append(j * dx)
        Plot_Xs.append((i + 1) * dx); Plot_Ys.append((j + 1) * dx)
        Plot_Xs.append(i * dx); Plot_Ys.append((j + 1) * dx)

        tris.append([base_v, base_v + 1, base_v + 2])
        tris.append([base_v, base_v + 2, base_v + 3])

        if i != M - 1:
            tris.append([base_v + 1, base_v + 4, base_v + 7])
            tris.append([base_v + 1, base_v + 7, base_v + 2])
        if j != N - 1:
            tris.append([base_v + 2, base_v + 1 + 4 * M, base_v + 4 * M])
            tris.append([base_v + 2, base_v + 4 * M, base_v + 3])
mesh_triangles = np.array(tris)

Plot_Xs = np.asarray(Plot_Xs)
Plot_Ys = np.asarray(Plot_Ys)
grid_centers = np.asarray(grid_centers)

masks = np.zeros([len(tris)])

np.set_printoptions(threshold=sys.maxsize, linewidth=np.nan)
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

# Simulation loop
def F(h, uv, z):
    return uv * uv / 2 + g * (h + z)

def preserve_volume(H_Original, U, V, dt):
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

vis = o3d.visualization.VisualizerWithKeyCallback()

vis.create_window()
stage_mesh = o3d.geometry.TriangleMesh()

stage_mesh.vertices = o3d.utility.Vector3dVector(np.array([[2, 2, 0],
                          [5, 2, 0],
                          [5, 5, 0]]))
stage_mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2]]).astype(np.int32))
stage_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([[0.1, 0.2, 1],
                          [0.1, 0.2, 1],
                          [0.1, 0.2, 1]]))
stage_mesh.compute_vertex_normals()

u_points = np.zeros([M * N * 2, 3])
v_points = np.zeros([M * N * 2, 3])
uv_lines = [[a, a + 1] for a in range(0, M * N * 2, 2)]
r_colors = [[1, 0, 0] for i in range(len(uv_lines))]
y_colors = [[1, 1, 0] for i in range(len(uv_lines))]
u_line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(u_points), lines=o3d.utility.Vector2iVector(uv_lines))
u_line_set.colors = o3d.utility.Vector3dVector(r_colors)
v_line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(v_points), lines=o3d.utility.Vector2iVector(uv_lines))
v_line_set.colors = o3d.utility.Vector3dVector(y_colors)
vis.add_geometry(u_line_set)
vis.add_geometry(v_line_set)

vis.add_geometry(stage_mesh)

single_step_mode = False
continuous_running = False
single_step_left = 0

def step_simulation(vis):
    global single_step_left

    if single_step_mode:
        if single_step_left > 0:
            single_step_left -= 1
        else:
            return
    else:
        if not continuous_running:
            return

    global step, H, U, V, dt

    step += 1

    if step % visual_interval == 0:
        stage = H[1:-1, 1:-1] + Z[1:-1, 1:-1]
        repeated_stage = np.transpose(np.vstack([stage.ravel()] * 4))

        mesh_vertices = np.transpose(np.vstack([Plot_Xs.ravel(), Plot_Ys.ravel(), repeated_stage.ravel()]))

        dry = H[1:-1, 1:-1] < dry_threshold_disp
        dry = np.reshape(dry, [-1])
        quad_colors = np.empty([len(dry), 3])
        quad_colors[:] = [0.1, 0.4, 0.9]
        quad_colors[np.where(dry)] = [0.7, 0.3, 0.2]
        vertex_colors = np.repeat(quad_colors, 4, axis=0)

        stage_mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices * 0.1)
        stage_mesh.triangles = o3d.utility.Vector3iVector(mesh_triangles)
        stage_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        stage_mesh.compute_vertex_normals()
        stage_mesh.compute_triangle_normals()

        grid_centers[:, 2] = stage.ravel()
        u_points[::2,:] = grid_centers
        u_points[1::2,:] = grid_centers + np.expand_dims(U[1:-1,1:-1].flatten(), axis=1) * np.vstack([np.asarray([0, 1, 0])] * M * N)
        v_points[::2,:] = grid_centers
        v_points[1::2,:] = grid_centers + np.expand_dims(V[1:-1,1:-1].flatten(), axis=1) * np.vstack([np.asarray([1, 0, 0])] * M * N)
        u_line_set.points = o3d.utility.Vector3dVector(u_points * 0.1)
        v_line_set.points = o3d.utility.Vector3dVector(v_points * 0.1)

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

    H3, Flow_X, Flow_Y = preserve_volume(H[1:-1, 1:-1], U2, V2, dt)
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
    H_Diff = np.sum(np.abs(H3 - H[1:-1, 1:-1]))
    H[1:-1, 1:-1] = H3
    U[1:-1, 1:-1] = U2
    V[1:-1, 1:-1] = V2

    if step % visual_interval == 0:
        title = '#%d Diff=%.3f V=%.2f MaxU=%.1f MaxV=%.1f dt=%.4f' % \
                (step, H_Diff, np.sum(H[1:-1, 1:-1]), np.max(np.abs(U)), np.max(np.abs(V)), dt)
        print(title)
        return True
    else:
        return False

def pause_resume(vis):
    global continuous_running, single_step_mode
    single_step_mode = False
    continuous_running = not continuous_running
    return False

def single_step(vis):
    if not continuous_running:
        global single_step_mode, single_step_left
        single_step_mode = True
        single_step_left = visual_interval


vis.register_animation_callback(step_simulation)
vis.register_key_callback(ord('A'), pause_resume)
vis.register_key_callback(ord('S'), single_step)

vis.run()
vis.destroy_window()
