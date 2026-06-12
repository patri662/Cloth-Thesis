# Need to restart kernel if changes done in the .py importations!!

import sys,os
notebook_dir = os.getcwd()  
parent_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
print(parent_dir)
sys.path.append(parent_dir)
# from implementation.Cloth import Cloth 
from implementation.Cloth_Ball import Cloth # need to call new code, which is in a new file!
from implementation.utils import createRectangularMesh, duplicate_node_pairs
import time
import numpy as np
import polyscope as ps


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

# solver parameters
dt = 1/60 #frame rate
tol = 0.0095 # up to 0.75% of relative error in constraint satisfaction to stop iterations

clothilde.preparePolyscope()

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

# rotate around Z-axis during simulation (baseline-style)
u = clothilde.positions[inds_ctr].copy()
# rotation center (pivot)
pivot = np.mean(u, axis=0)
# total rotation angle
theta_total = np.deg2rad(-30)
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

clothilde.plotMesh()

# rotate around Y-axis + open during simulation 
tf = 40
u = clothilde.positions[inds_ctr].copy()

left_mask = np.array(inds_ctr) < na*nb
right_mask = ~left_mask

theta_total = np.deg2rad(-20)
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


    clothilde.simulate(u=u, control=inds_ctr)

# leave it there
u = clothilde.positions[inds_ctr].copy()
for i in range(200):
    clothilde.simulate(u=u, control=inds_ctr)

clothilde.plotMesh()


# Save video of simulation
def save_frames_ps(history_pos, Am, label, folder="frames", step=4):
    os.makedirs(folder, exist_ok=True)
    n_frames = len(history_pos)
    print(f"Saving {n_frames//step} frames to {folder}/")

    for i, frame_idx in enumerate(range(0, n_frames, step)):
        phi_mat = history_pos[frame_idx]
        phi_all = Am @ phi_mat

        ps.get_surface_mesh(label).update_vertex_positions(phi_all)
        ps.screenshot(os.path.join(folder, f"frame_{i:04d}.png"), transparent_bg=False)


        if i % 50 == 0:
            print(f"Saved frame {i} / {n_frames//step}")

import subprocess
import shutil

def create_video_from_frames(frames_folder, output_video, framerate=20, cleanup = True):
    """
    Create a video from a sequence of frames using ffmpeg.
    
    Args:
        frames_folder: Path to folder containing frame_XXXX.png files
        output_video: Output video filename (e.g., "experiment.mp4")
        framerate: Framerate for the video (default: 20)
        cleanup: If True, delete the frames folder after video creation (default: True)
    """
    try:
        cmd = [
            "ffmpeg",
            "-framerate", str(framerate),
            "-i", f"{frames_folder}/frame_%04d.png",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-y",  # overwrite output file without asking
            output_video
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Video saved: {output_video}")

        # Clean up frames folder if requested
        if cleanup:
            shutil.rmtree(frames_folder)
            print(f"Removed frames folder: {frames_folder}")

    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e.stderr.decode()}")
    except FileNotFoundError:
        print("ffmpeg not found. Install it using: brew install ffmpeg")
    except Exception as e:
        print(f"Error during cleanup: {e}")

# Save frames and create video
frames_folder = f"Bag-Ball Manipulation Frames Delete"
os.makedirs(frames_folder, exist_ok=True)

# Fix camera view
ps.reset_camera_to_home_view()
ps.set_view_projection_mode("perspective")
ps.set_window_size(1500, 1000)

intrinsics = ps.CameraIntrinsics(fov_vertical_deg=45., fov_horizontal_deg=45.)

angle = np.radians(90)
radius = 5
height = 1

# Compute camera position
cam_x = radius * np.sin(angle)   # horizontal offset
cam_z = height                    # vertical offset (low, close to table)
cam_y = radius * np.cos(angle)    # forward/back offset

# Look at origin
look_dir = (-cam_x, -cam_y, -cam_z)
up_dir = (0, 0, 1) # with the mesh rotated, now the Y-axis is the original Z-axis, so we set up to be along global Z

extrinsics = ps.CameraExtrinsics(root=(cam_x, cam_y, cam_z), look_dir=look_dir, up_dir=up_dir)

ps.set_view_camera_parameters(ps.CameraParameters(intrinsics, extrinsics))

# edges configuration
label = clothilde.label
mesh = ps.get_surface_mesh(label)

mesh.set_color((0.2, 0.5, 0.8)) # RGB color of the mesh, blue for all meshes/cases (if not determined, each mesh for each new experiment will have different colors when polyscope registers them)
mesh.set_edge_color((0.0, 0.0, 0.0))  # RGB color of the edges, black for edges
mesh.set_edge_width(0.9)             # thickness of the edges

# change view here and will save as here!
clothilde.makeMovie(1,False,0) # True/False to repeat simulation when finishes

save_frames_ps(clothilde.history_pos, clothilde.Am, 
                clothilde.label, folder=frames_folder, step=4)
create_video_from_frames(frames_folder, "Bag Manipulation.mp4", cleanup=False) # no deleting frames for report