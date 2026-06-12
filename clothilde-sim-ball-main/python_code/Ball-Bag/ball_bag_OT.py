# Need to restart kernel if changes done in the .py importations!!

import sys,os
notebook_dir = os.getcwd()  
parent_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
print(parent_dir)
sys.path.append(parent_dir)
# from implementation.Cloth import Cloth 
from implementation.Cloth_Ball import Cloth # need to call new code, which is in a new file!
from implementation.utils import createRectangularMesh, duplicate_node_pairs
import numpy as np

try:
    import ot as pot
    _POT = True
except ImportError:
    _POT = False
    
RESULTS_PATH = "release_cloud.npy"

def create_rect_mesh(a=1, b=1): # the difference between a and b cannot be double!!
  # Dimensions need to be feasible to make two meshes join
    # a cannot be smaller than b and take into account how curved the mesh is with h param.
    # a is the width, so needs to be long enough to make edges join (at least a = 1)

    # need to form squares more or less depending on a and b!!
    # tried na = 20 first, but simulation very slow so needed to reduce number of nodes (less precise but no worries)!!
    na = int(a/0.05) # 1/10 = 0.1 cm/node -> so for a : a/0.1= 1cm/0.1 = 10 nodes
    nb = int(b/0.05) # then, same for b: b/0.1 to form squares (but integer, no decimals)

    print('Horizontal nodes:', na)
    print('Vertical nodes:', nb)

    np.random.seed(10)
        
    X, T = createRectangularMesh(a=a, b=b, na=na, nb=nb, h=0.80)

    return X, T, na, nb

# Define mesh with progressive flattening along vertical (along b)

X, T, na, nb = create_rect_mesh()

# Exponential Decay Function
X[:,2] = (1 - np.exp(3*(X[:,1]-0.5)))*X[:,2]

# The substracting element b/2 is the center point of the mesh, 
# so need to define exp. function around this center in order to make both meshes join at the bottom edge
# If center point wrong, the bottom edges don't match and flattening will happen around center at that element
# The value that multiplies in the exp() controls how fast curve changes, so smaller = flatter
# and the 1-... makes it decaying
# b = 1.6
# X[:,2] = (1 - np.exp(0.3*(X[:,1]-b/2)))*X[:,2]

print('Shape for X vecor:', X.shape)

# Make copy of the initial mesh to form the to halfs and join them with the seams

Y = X.copy(); # duplication of mesh

# since we rotated the initial mesh by 90º to have it vertical,
# the points that need to be inversed are now in Z-axis in position 1 
# (not in position 2 as original one w/o rotating)

Y[:,2] = -Y[:,2] # rotate initial mesh by 180º to be opposite to initial one
                # to have () meshes, not (( meshes, so edges join with seams

# join all nodes positions into unique mesh, although two different cloth pieces will be there before joined
X = np.concatenate([X,Y]); T = np.concatenate([T,T+na*nb])


# Rotate initial rectangular mesh -90° around X-axis
theta = -np.pi / 2
Rx = np.array([[1, 0, 0],
               [0, np.cos(theta), -np.sin(theta)],
               [0, np.sin(theta), np.cos(theta)]])
X = X @ Rx.T

seam = duplicate_node_pairs(X)
seam = np.sort(seam, axis=0)
seam = seam[36:, :]
print('Shape of the custom seams matrix:', seam.shape)

X[:,2] += 0.80 

clothilde = Cloth(X, T, seam); 
# clothilde.plotMesh()

# initialize ball inside Cloth class by caling method inside cloth
clothilde.ball = clothilde.addBall(position=[0.0,0.0,1.3], rad=0.06, mass=0.2, friction=0.0)
clothilde.preparePolyscope()
clothilde.ball.plotMesh()


def run_simulation(theta_1=-45, theta_2=-40, tf_2=52):

    X, T, na, nb = create_rect_mesh()

    # Exponential Decay Function
    X[:,2] = (1 - np.exp(3*(X[:,1]-0.5)))*X[:,2]

    Y = X.copy(); # duplication of mesh

    Y[:,2] = -Y[:,2] 

    X = np.concatenate([X,Y]); T = np.concatenate([T,T+na*nb])

    # Rotate initial rectangular mesh -90° around X-axis
    theta = -np.pi / 2
    Rx = np.array([[1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]])
    X = X @ Rx.T

    seam = duplicate_node_pairs(X)
    seam = np.sort(seam, axis=0)
    seam = seam[36:, :]

    X[:,2] += 0.80 

    clothilde = Cloth(X, T, seam); 

    # initialize ball inside Cloth class by caling method inside cloth
    clothilde.ball = clothilde.addBall(position=[0.0,0.0,1.3], rad=0.06, mass=0.2, friction=0.0)

    # solver parameters
    dt = 1/60 #frame rate
    tol = 0.0095 # up to 0.75% of relative error in constraint satisfaction to stop iterations

    clothilde.preparePolyscope()
    # clothilde.ball.plotMesh()

    #physical parameters
    rho = 0.1 #cloth density
    delta = 0.08 # aerodynamics parameter: between 0 and rho
    kappa = 0.15*1e-4 # stifness or bending resistance
    alpha = 0.2 #damping of oscillations
    shr = 5*1e-4 #allowed shearing resistance
    strh = 0.01*1e-4 #allowed stretching resistance
    mu_f = 0.25 #friction with the floor
    mu_s = 0.4 #friction with the cloth itself
    thck = 0.95 #size of the balls
    sub_steps = 12 #numer of intermidate steps between each dt


    clothilde.setSimulatorParameters(dt=dt,tol=tol,sub_steps = sub_steps,
                                rho=rho,delta=delta,kappa=kappa,shr=shr,
                                str=strh,alpha=alpha,mu_f=mu_f,mu_s=mu_s,
                                thck=thck)

    clothilde.restart()

    # define the control line for each half of the bag -> take half of na centered nodes for each
    first_mesh = list(range(int(na/2-5), int(na/2+5))) # 1st mesh nodes start at 0, so take from 5 to 15 to have 5 centered nodes
    second_mesh = list(range(na*nb + int(na/2-5), na*nb + int(na/2+5))) # 2nd mesh nodes start at end of 1rst mesh node indexes
    inds_ctr = [0,na-1]
    inds_ctr = first_mesh + second_mesh

    # only grasp it and stabilize it
    u = clothilde.positions[inds_ctr].copy()
    for _ in range(200):
        clothilde.simulate(u=u, control=inds_ctr)

    print('Initial ball position:', clothilde.ball.position)

    # Phase 1 (theta_1)
    # rotate around Z-axis during simulation (baseline-style)
    u = clothilde.positions[inds_ctr].copy()
    # rotation center (pivot)
    pivot = np.mean(u, axis=0)
    # total rotation angle
    theta_total = np.deg2rad(theta_1)
    # number of simulation steps
    nsteps = 10

    for i in range(nsteps):
        # increasing angle -> angular velocity
        dtheta = theta_total * (i + 1) / nsteps

        # rotation matrix around Z axis
        Rz = np.array([
            [np.cos(dtheta), -np.sin(dtheta), 0],
            [np.sin(dtheta), np.cos(dtheta), 0],
            [0, 0, 1]
        ])

        # rotate around pivot
        centered = u - pivot
        rotated = centered @ Rz.T + pivot

        clothilde.simulate(u=rotated, control=inds_ctr)

    # only grasp it and stabilize it
    u = clothilde.positions[inds_ctr].copy()
    for _ in range(100):
        clothilde.simulate(u=u, control=inds_ctr)

    # rotate around Y-axis + open during simulation 
    tf = tf_2
    u = clothilde.positions[inds_ctr].copy()

    left_mask = np.array(inds_ctr) < na*nb
    right_mask = ~left_mask

    theta_total = np.deg2rad(theta_2)
    dist_total = 0.80

    for i in range(tf):

        # --- LOCAL FRAME ---
        left_center = u[left_mask].mean(axis=0)
        right_center = u[right_mask].mean(axis=0)

        local_x = right_center - left_center
        local_x /= np.linalg.norm(local_x)

        world_up = np.array([0, 0, 1])

        local_y = np.cross(world_up, local_x)
        local_y /= np.linalg.norm(local_y)

        local_z = np.cross(local_x, local_y)
        local_z /= np.linalg.norm(local_z)

        # opening level per step
        ddist = dist_total / tf

        # --- OPENING ---
        u[left_mask] -= ddist * local_x
        u[right_mask] += ddist * local_x

        # --- ROTATION  OF WHOLE BAG AROUND LOCAL Y ---
        dtheta = theta_total / tf

        ux, uy, uz = local_y

        # Rodrigues rotation matrix
        c = np.cos(dtheta)
        s = np.sin(dtheta)

        R = np.array([
            [c + ux**2*(1-c), ux*uy*(1-c)-uz*s, ux*uz*(1-c)+uy*s],
            [uy*ux*(1-c)+uz*s, c+uy**2*(1-c), uy*uz*(1-c)-ux*s],
            [uz*ux*(1-c)-uy*s, uz*uy*(1-c)+ux*s, c+uz**2*(1-c)]
        ])

        # rotate around bag center
        pivot = (left_center + right_center) / 2

        centered = u - pivot
        u = centered @ R.T + pivot

        # try by also sweeping these rotation and its speed!!
        # # ROTATION OF EACH GRIP SURFACE AROUND Y TO DIRECTION BALL
        # local_theta = np.deg2rad(10) / tf

        # # rotate left and right 
        # # if same direction , then same sign, if opposite, then different signs to rotate in opposite directions
        # for mask, sign in [(left_mask, +1), (right_mask, +1)]:

        #     grip_pts = u[mask]

        #     # local pivot of this hand
        #     grip_pivot = grip_pts.mean(axis=0)

        #     # rotation around current local_y axis
        #     ux, uy, uz = local_x

        #     c = np.cos(sign * local_theta)
        #     s = np.sin(sign * local_theta)

        #     R_local = np.array([
        #         [c + ux**2*(1-c), ux*uy*(1-c)-uz*s, ux*uz*(1-c)+uy*s],
        #         [uy*ux*(1-c)+uz*s, c+uy**2*(1-c), uy*uz*(1-c)-ux*s],
        #         [uz*ux*(1-c)-uy*s, uz*uy*(1-c)+ux*s, c+uz**2*(1-c)]
        #     ])

        #     centered_grip = grip_pts - grip_pivot
        #     rotated_grip = centered_grip @ R_local.T + grip_pivot

        #     u[mask] = rotated_grip


        clothilde.simulate(u=u, control=inds_ctr)

    release_speed = clothilde.ball.velocity.copy() 

    print(f"Ball speed per coord.: {clothilde.ball.velocity}, Ball speed: {np.linalg.norm(clothilde.ball.velocity)}")
    # print(f"Ball position after opening: {clothilde.ball.position}") # still touched by bag

    # leave it there
    traj = [] # record free ball trajectory after opening, but not used for now, so reset to empty list to save memory
    u = clothilde.positions[inds_ctr].copy()
    for i in range(200):
        clothilde.simulate(u=u, control=inds_ctr)
        traj.append(clothilde.ball.position.copy())
        # # save ball position while not on floor
        # if clothilde.ball.position[2] > 1.2*clothilde.ball.rad: # only save while ball is in the air, not while in the floor (same threshold as Ball class)
        #     traj.append(clothilde.ball.position.copy())
        # else:
        #     break
        # not break buit stop saving traj maybe? because sim stops then

    traj = np.array(traj)

    print(f"Ball position after release: {traj[0]}")
    print('Final ball position:', clothilde.ball.position)

    # clothilde.makeMovie(1,False,0)

    return {
        'theta_1': theta_1,
        'theta_2': theta_2,
        'tf_2': tf_2,
        'ball_velocity': release_speed.copy(),
        'ball_linear_speed': np.linalg.norm(release_speed),
        'ball_trajectory': traj.copy(),
        'ball_release_position': traj[0].copy(),
        'ball_final_position': clothilde.ball.position.copy(),
    }

# Make goal a cloud of points!!

def make_goal_cloud(goal_pos, n=30, radius=0.1, seed=0):
    rng = np.random.default_rng(seed)
    goal_pos = np.array(goal_pos, float)
    pts = goal_pos + rng.normal(scale=radius, size=(n, 3))
    pts[:, 2] = goal_pos[2]
    return pts

# ─────────────────────────────────────────────────────────────────────
# OT cost
# ─────────────────────────────────────────────────────────────────────

def ot_cost(source_pts, target_pts):
    s = np.atleast_2d(source_pts)
    t = np.atleast_2d(target_pts)
    if _POT and len(s) and len(t):
        a = np.ones(len(s)) / len(s)
        b = np.ones(len(t)) / len(t)
        C = pot.dist(s, t)
        T = pot.emd(a, b, C)
        return float(np.sqrt(np.maximum(np.sum(T * C), 0.0)))
    return float(np.linalg.norm(s.mean(0) - t.mean(0)))

# try different parameters for opening and rotation
theta_1_vals   = [-60,-45,-30,-15,0]
theta_2_vals   = [-60,-50,-40,-30,-20]
tf_2 = [20, 30, 40, 50, 60]

# save sweeping results if not saved already
if os.path.exists(RESULTS_PATH):
    # Load existing results, skip simulation
    release_cloud = list(np.load(RESULTS_PATH, allow_pickle=True))
    print(f"Loaded {len(release_cloud)} existing results.")
else:
    release_cloud = []
    for theta_1 in theta_1_vals:
        for theta_2 in theta_2_vals:
            for tf in tf_2:
                print(f"\nRunning simulation with theta_1={theta_1}, theta_2={theta_2}, tf={tf}")
                result =run_simulation(theta_1=theta_1, theta_2=theta_2, tf_2=tf)
                release_cloud.append(result)
    np.save(RESULTS_PATH, np.array(release_cloud, dtype=object))
    print(f"Saved {len(release_cloud)} results to {RESULTS_PATH}.")

# OT
# the Z goal is the position of the center of the ball, which is the threshold for floorCollisions (1.2*ball.rad)
goal = np.array([1.0, 1.0, 0.072]) # chosen based on manipulation given

# L2 error baseline
for sample in release_cloud:
    landing = sample["ball_final_position"]
    sample["error"] = np.linalg.norm(landing - goal)

best = min(
    release_cloud,
    key=lambda s: s["error"])
print(f"Best sample (L2 error): {best}")

# best manip with OT
goal_cloud = make_goal_cloud(goal,n=30,radius=0.3)

for sample in release_cloud:
    landing = sample["ball_final_position"]
    sample["W2"] = ot_cost(landing[np.newaxis,:], goal_cloud)

best_ot = min(
    release_cloud,
    key=lambda s: s["W2"])

print("Best OT sample:")
print(best_ot)

# whole cloud OT cost
landing_cloud = np.array([
    s["ball_final_position"]
    for s in release_cloud])

best_cloud = ot_cost(
    landing_cloud,
    goal_cloud)

print(f"Landing cloud mean: {landing_cloud.mean(axis=0)}")
print(f"Goal cloud mean: {goal_cloud.mean(axis=0)}")

centroid_error = np.linalg.norm(landing_cloud.mean(axis=0) - goal_cloud.mean(axis=0))

print("Centroid error:", centroid_error)
print(f"Best sample (W2 error): {best_cloud}")


# Try different goals and compare results for each of them

# Load your saved results
release_cloud = list(np.load("release_cloud.npy", allow_pickle=True))

# Define 4 goals -> positions that make physical sense for setup
goals = {
    "G1": np.array([ 1.0,  1.0, 0.072]),   # original
    "G2": np.array([ 0.5,  1.5, 0.072]),   # shifted left/forward
    "G3": np.array([ 1.5,  0.5, 0.072]),   # shifted right/back
    "G4": np.array([ 0.0,  0.0, 0.072]),   # shifted right/back
}

landing_cloud = np.array([s["ball_final_position"] for s in release_cloud])

for name, g in goals.items():
    goal_cloud = make_goal_cloud(g, n=30, radius=0.3)

    # L2 error baseline
    for sample in release_cloud:
        landing = sample["ball_final_position"]
        sample[f"L2_{name}"] = np.linalg.norm(landing - g)

    best_L2 = min(release_cloud, key=lambda s: s[f"L2_{name}"])

    # Per-sample W2
    for sample in release_cloud:
        landing = sample["ball_final_position"]
        sample[f"W2_{name}"] = ot_cost(landing[np.newaxis,:], goal_cloud)
    
    best = min(release_cloud, key=lambda s: s[f"W2_{name}"])

    # whole cloud OT cost
    global_w2 = ot_cost(landing_cloud, goal_cloud)
    centroid_err = np.linalg.norm(landing_cloud.mean(0) - goal_cloud.mean(0))
    
    print(f"\n--- {name}: goal={g} ---")
    print(f"  Best L2 params:     θ1={best_L2['theta_1']}, θ2={best_L2['theta_2']}, tf={best_L2['tf_2']}")
    print(f" Best L2 error: {best_L2[f'L2_{name}']:.4f} m")
    print(f"  Best W2 params:     θ1={best['theta_1']}, θ2={best['theta_2']}, tf={best['tf_2']}")
    print(f"  Best W2 error:    {best[f'L2_{name}']:.4f} m")
    print(f"  Best W2:         {best[f'W2_{name}']:.4f} m")
    print(f" Best W2 Landing: {best[f'ball_final_position']} m ")
    print(f"  Global W2:       {global_w2:.4f} m")
    print(f"  Centroid error:  {centroid_err:.4f} m")


# Draw landing cloud and goal clouds for each goal together

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, (name, g) in zip(axes, goals.items()):
    goal_cloud = make_goal_cloud(g, n=30, radius=0.3)
    
    ax.scatter(landing_cloud[:,0], landing_cloud[:,1],
               s=20, alpha=0.6, label="Landing cloud")
    ax.scatter(goal_cloud[:,0], goal_cloud[:,1],
               s=20, alpha=0.4, color="orange", label="Goal cloud")
    ax.scatter(*g[:2], marker="x", s=100, color="red", zorder=5, label="Goal centre")
    
    ax.set_title(f"{name}: g={g[:2]}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.legend(fontsize=7)

plt.suptitle("Landing Cloud vs Goal Cloud", fontsize=13)
plt.tight_layout()
plt.savefig("landing_vs_goals.png", dpi=150)
plt.show()

# FIGURE 1 — scatter plots with best-L2 and best-W2 markers

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Flatten the 2x2 array into a 1D list of 4 axes
axes = axes.ravel()

for ax, (name, g) in zip(axes, goals.items()):
    goal_cloud = make_goal_cloud(g, n=30, radius=0.3)

    # L2 error baseline
    for sample in release_cloud:
        landing = sample["ball_final_position"]
        sample[f"L2_{name}"] = np.linalg.norm(landing - g)

    best_L2 = min(release_cloud, key=lambda s: s[f"L2_{name}"])

    # Per-sample W2
    for sample in release_cloud:
        landing = sample["ball_final_position"]
        sample[f"W2_{name}"] = ot_cost(landing[np.newaxis,:], goal_cloud)
    
    best = min(release_cloud, key=lambda s: s[f"W2_{name}"])

    # whole cloud OT cost
    global_w2 = ot_cost(landing_cloud, goal_cloud)
    centroid_err = np.linalg.norm(landing_cloud.mean(0) - goal_cloud.mean(0))
 
    bl2 = best_L2[f'ball_final_position']
    bot = best[f'ball_final_position']
    same = np.allclose(bl2[:1], bot[:1], atol=1e-6)   # flag if they coincide
 
    # --- base scatter ---
    ax.scatter(landing_cloud[:, 0], landing_cloud[:, 1],
               s=18, alpha=0.45, color="#4c72b0", label="Landing Cloud", zorder=2)
    ax.scatter(goal_cloud[:, 0], goal_cloud[:, 1],
               s=18, alpha=0.35, color="orange", label="Goal Cloud", zorder=2)
    ax.scatter(*g[:2], marker="x", s=120, color="red",
               linewidths=2, zorder=6, label="Goal Centroid")
 
    # --- best L2 marker ---
    ax.scatter(*bl2[:2], marker="^", s=140, color="limegreen",
               edgecolors="black", linewidths=0.8, zorder=7,
               label=f"Best $L_2$  ({bl2[0]:.2f}, {bl2[1]:.2f})")
 
    # --- best OT marker (offset slightly if identical to L2) ---
    if same:
        # draw a single marker but mention both in the label
        ax.scatter(*bot[:2], marker="*", s=260, color="gold",
                   edgecolors="black", linewidths=0.8, zorder=8,
                   label=f"Best $W_2$ = Best $L_2$")
    else:
        ax.scatter(*bot[:2], marker="*", s=260, color="gold",
                   edgecolors="black", linewidths=0.8, zorder=8,
                   label=f"Best $W_2$  ({bot[0]:.2f}, {bot[1]:.2f})")
 
    ax.set_title(f"{name}:  $g$ = ({g[0]}, {g[1]}) m", fontsize=11)
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    ax.axis("equal")
    ax.legend(fontsize=6.5, loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)
 
plt.suptitle("Landing Cloud vs Goal Cloud",
             fontsize=13)
plt.tight_layout()
plt.savefig("landing_vs_goals_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: landing_vs_goals_results.png")
 
# FIGURE 2 — KDE heatmap of all 125 landing positions + goal centroids

from scipy.stats import gaussian_kde

# Build KDE over XY landings
xy = landing_cloud[:, :2].T                         # shape (2, 125)
kde = gaussian_kde(xy, bw_method="scott")
 
# Grid for evaluation — slightly padded around all data + goals
all_pts = np.vstack([landing_cloud[:, :2]] +
                    [g[:2][np.newaxis] for g in goals.values()])
x_min, y_min = all_pts.min(axis=0) - 0.3
x_max, y_max = all_pts.max(axis=0) + 0.3
 
grid_res = 300
xg = np.linspace(x_min, x_max, grid_res)
yg = np.linspace(y_min, y_max, grid_res)
Xg, Yg = np.meshgrid(xg, yg)
positions = np.vstack([Xg.ravel(), Yg.ravel()])
Z = kde(positions).reshape(grid_res, grid_res)
 
fig, ax = plt.subplots(figsize=(7, 6))
 
# heatmap
heatmap = ax.contourf(Xg, Yg, Z, levels=20, cmap="Blues", alpha=0.85)
cbar = fig.colorbar(heatmap, ax=ax, label="Probability density")
 
# scatter the raw landing positions on top (small, semi-transparent)
ax.scatter(landing_cloud[:, 0], landing_cloud[:, 1],
           s=12, color="steelblue", alpha=0.5, zorder=3, label="Landings")
 
# goal centroids with distinct markers and labels
goal_markers  = {"G1": "o", "G2": "s", "G3": "D", "G4": "^"}
goal_colors   = {"G1": "red", "G2": "darkorange", "G3": "purple", "G4": "crimson"}
 
for name, g in goals.items():
    ax.scatter(*g[:2],
               marker=goal_markers[name],
               s=160,
               color=goal_colors[name],
               edgecolors="white",
               linewidths=1.2,
               zorder=6,
               label=f"{name} = ({g[0]}, {g[1]}) m")
 
ax.set_xlabel("$x$ (m)", fontsize=12)
ax.set_ylabel("$y$ (m)", fontsize=12)
ax.set_title("Density of 125 landing positions\nwith the goal cloud centroids", fontsize=13)
ax.axis("equal")
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, linestyle="--", alpha=0.3)
 
plt.tight_layout()
plt.savefig("landing_kde_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: landing_kde_heatmap.png")

# FIGURE 3 — ball trajectory for each best case

fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': '3d'})

# Flatten the 2x2 array into a 1D list of 4 axes
axes = axes.ravel()

for ax, (name, g) in zip(axes, goals.items()):
    goal_cloud = make_goal_cloud(g, n=30, radius=0.3)

    # Per-sample W2
    for sample in release_cloud:
        landing = sample["ball_final_position"]
        sample[f"W2_{name}"] = ot_cost(
            landing[np.newaxis, :],
            goal_cloud
        )

    best = min(release_cloud, key=lambda s: s[f"W2_{name}"])
    traj = best["ball_trajectory"]

    ax.plot(
        traj[:, 0],
        traj[:, 1],
        traj[:, 2],
        linewidth=2
    )

    # start point
    ax.scatter(
        traj[0, 0],
        traj[0, 1],
        traj[0, 2],
        s=80,
        label="start"
    )

    # end point
    ax.scatter(
        traj[-1, 0],
        traj[-1, 1],
        traj[-1, 2],
        s=80,
        label="end"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{name}:  $g$ = ({g[0]}, {g[1]}) m", fontsize=11)
    ax.legend()

plt.suptitle("Ball Trajectory for each Best Landing", fontsize=13)
plt.tight_layout()
plt.savefig("ball_traj_OT.png", dpi=150, bbox_inches="tight")
plt.show()