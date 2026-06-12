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

na = 30; nb = 18
X, T = createRectangularMesh(a = 0.9,b = 0.5, na = na, nb = nb, h = 0.15)
X[:,2] += 0.4; #adjust height

clothilde = Cloth(X, T)

clothilde.ball = clothilde.addBall(position=[0.3,0.0,0.6], rad=0.06, mass=0.2, friction=0.3)

print(f'Ball position: {clothilde.ball.position}')   # position
print(f'Ball speed: {clothilde.ball.velocity}')      # velocity
print(f'Ball radius: {clothilde.ball.rad}')    # radius
print(f'Ball friction: {clothilde.ball.mu_b}')     # friction
print(f'Ball position history: {clothilde.ball.history_pos}')  # full trajectory


# solver parameters
dt = 1/60 #frame rate
tol = 0.009 # up to 0.75% of relative error in constraint satisfaction to stop iterations

#physical parameters
rho = 0.1 #cloth density
delta = 0.1 # aerodynamics parameter: between 0 and rho
kappa = 0.35*1e-4 # stifness or bending resistance
kappa_bnd = 0.015*1e-4 # stifness or bending resistance

alpha = 0.3 #damping of oscillations
shr = 1*1e-4 #allowed shearing resistance
strh = 0.001*1e-4 #allowed stretching resistance
mu_f = 0.45 #friction with the floor
mu_s = 0.4 #friction with the cloth itself
thck = 0.95 #size of the balls
sub_steps = 12 #number of intermidate steps between each dt

clothilde.setSimulatorParameters(dt=dt,tol=tol,sub_steps = sub_steps,
                                rho=rho,delta=delta,kappa=kappa,kappa_bnd=kappa_bnd,shr=shr,
                                str=strh,alpha=alpha,mu_f=mu_f,mu_s=mu_s,
                                thck=thck)

clothilde.preparePolyscope()
clothilde.ball.plotMesh()

inds_ctr = [0,na-1, (na*nb)-na, (na*nb)-1]
tf = int(6/dt)
u = X[inds_ctr]
for i in range(tf):
    clothilde.simulate(u = u, control = inds_ctr)

print('Average iterations',clothilde.total_iters/(len(clothilde.history_pos)-1))
clothilde.makeMovie(1,False,2) # True/False to repeat simulation when finishes

# Save video of simulation
def save_frames_ps(history_pos, history_pos_ball, Am, label, label_ball, folder="frames", step=4):
    os.makedirs(folder, exist_ok=True)
    n_frames = len(history_pos)
    print(f"Saving {n_frames//step} frames to {folder}/")

    for i, frame_idx in enumerate(range(0, n_frames, step)):
        # update cloth
        phi_mat = history_pos[frame_idx]
        phi_all = Am @ phi_mat
        ps.get_surface_mesh(label).update_vertex_positions(phi_all)

        # update ball
        ball_pos = history_pos_ball[frame_idx]
        phi_ball = ball_pos[np.newaxis, :]
        ps.get_point_cloud(label_ball).update_point_positions(phi_ball)

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

save_frames_ps(clothilde.history_pos, clothilde.ball.history_pos, clothilde.Am, 
                clothilde.label, clothilde.ball.label, folder=frames_folder, step=4)
create_video_from_frames(frames_folder, "Bag-Ball Friction.mp4", cleanup=True) 