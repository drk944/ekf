import numpy as np 
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Ellipse
import matplotlib.lines as mlines
from tqdm import tqdm
# import jax
# import jax.numpy as jnp
# from jax import jit, vmap
import time
# jax.config.update('jax_platform_name', 'cpu')

# robot association is an array where each row corresponds a bar code value to a landmark
robot_association = np.loadtxt('datasets/MRCLAM_Dataset1/Barcodes.dat', skiprows=4, dtype='int')
# create empy dictionary
robot_association_dict = {}
# iterate through each row
for row in robot_association:
    # add the key value pair to the dictionary
    robot_association_dict[row[1]] = int(row[0])

# Landmark ground truth (where are the landmarks in the world)
landmark_gt = np.loadtxt('datasets/MRCLAM_Dataset1/Landmark_Groundtruth.dat', skiprows=4, dtype='float')
landmark_dict = {}
for row in landmark_gt:
    landmark_dict[int(row[0])] = row[1:3]

# Robot 1 ground truth
robot_1_gt = np.loadtxt('datasets/MRCLAM_Dataset1/Robot1_Groundtruth.dat', skiprows=4, dtype='float')
# Robot measurements. Each row has a timestep, a landmark ID, and a measurement in the form of range (m) and bearing (rad)
robot1_measurements = np.loadtxt('datasets/MRCLAM_Dataset1/Robot1_Measurement.dat', skiprows=4, dtype='float')

# remove rows in robot1 measurement that correspond to other robots and not landmarks
# This data set was designed for cooperative SLAM but we only care about the single agent Robot 1
robot1_measurements = robot1_measurements[robot1_measurements[:,1] != 14]
robot1_measurements = robot1_measurements[robot1_measurements[:,1] != 41]
robot1_measurements = robot1_measurements[robot1_measurements[:,1] != 32]
robot1_measurements = robot1_measurements[robot1_measurements[:,1] != 23]
robot1_measurements = robot1_measurements[robot1_measurements[:,1] != 61] # 17 is dumb
robot1_measurements = robot1_measurements[robot1_measurements[:,1] != 18] # 11 is also dumb
# get only the unique lines based on the first column
unique_timesteps, idx = np.unique(robot1_measurements[:, 0], return_index=True)
# extract rows with unique timesteps
robot1_measurements = robot1_measurements[idx]

# Robot odometry is recorded as: timestep, forward velocity (m/s), angular velocity (rad/s)
robot1_odometry = np.loadtxt('datasets/MRCLAM_Dataset1/Robot1_Odometry.dat', skiprows=4, dtype='float')

def dead_reckoning(robot1_odometry, initial_x, initial_y, initial_theta):
    x = initial_x # We need to know where the robot is starting from
    y = initial_y
    theta = initial_theta

    positions = np.empty((robot1_odometry.shape[0], 3))
    # Add the initial position to the array
    positions[0] = [x, y, theta]

    for i in range(1, robot1_odometry.shape[0]):
        time_step = robot1_odometry[i,0] - robot1_odometry[i-1,0]
        velocity = robot1_odometry[i,1]
        angular_velocity = robot1_odometry[i,2]
        x += velocity * math.cos(theta) * time_step
        y += velocity * math.sin(theta) * time_step
        theta += angular_velocity * time_step
        positions[i] = [x, y, theta]
    return positions

def motion_mu_bar(mu_bar, u, dt):
    vt = u[0]
    wt = u[1]

    # u is the odometry
    # mu_bar is the predicted mean
    mu_bar[0] += vt * dt * math.cos(mu_bar[2]+((wt*dt)/2))
    mu_bar[1] += vt * dt * math.sin(mu_bar[2]+((wt*dt)/2))
    mu_bar[2] += wt*dt
    return mu_bar

def motion_sigma_bar(sigma_bar, mu_bar, u, dt, alphas):
    vt = u[0]
    wt = u[1]

    g_t = np.array([[1, 0, -vt*dt*np.sin(mu_bar[2]+((wt*dt)/2))],
                    [0, 1, vt*dt*np.cos(mu_bar[2]+((wt*dt)/2))],
                    [0, 0, 1]])
    
    m_t = np.array([[alphas[0]*vt**2 + alphas[1]*wt**2, 0],
                    [0, alphas[2]*vt**2 + alphas[3]*wt**2]])
    
    v_t = np.array([[np.cos(mu_bar[2]+((wt*dt)/2)), -0.5*np.sin(mu_bar[2]+((wt*dt)/2))],
                    [np.sin(mu_bar[2]+((wt*dt)/2)), 0.5*np.cos(mu_bar[2]+((wt*dt)/2))],
                    [0, 1]])

    sigma_bar = (g_t @ sigma_bar @ g_t.T) + (v_t @ m_t @ v_t.T)

    return sigma_bar

sigma_range = 2.5
sigma_bearing = 5
sigma_id = 1

Q_t = [[sigma_range**2, 0, 0,],
       [0, sigma_bearing**2, 0],
       [0, 0, sigma_id**2]]

# alphas = [.04, .0009, .0081, .0064] # robot-dependent motion noise 
# alphas = [.0010, .00001, .0010, .00001] # robot-dependent motion noise 
alphas = [.0009, .000008, .0009, .000008] # robot-dependent motion noise 


def sensor_reading(mu_bar, sigma_bar, z, landmarks):
    landmark_id = robot_association_dict[int(z[0])]
    landmark = landmark_dict[landmark_id]
    q = (landmark[0] - mu_bar[0])**2 + (landmark[1] - mu_bar[1])**2
    z_hat = np.array([math.sqrt(q), math.atan2(landmark[1] - mu_bar[1], landmark[0] - mu_bar[0]) - mu_bar[2]])
    
    h_t = np.array([[-(landmark[0] - mu_bar[0])/math.sqrt(q), -(landmark[1] - mu_bar[1])/math.sqrt(q), 0],
                    [(landmark[1] - mu_bar[1])/q, - (landmark[0] - mu_bar[0])/q, -1],
                    [0, 0, 0]])
    
    S_t = h_t @ sigma_bar @ h_t.T + Q_t 

    K_t = sigma_bar @ h_t.T @ np.linalg.inv(S_t)  # Kalman Gain, inverting a matrix, will look to optimize this later
    
    z_hat[-1] -= math.pi/2
    z_diff = z[1:] - z_hat
    # add a 0 to the end of the z_diff to account for the id
    z_diff = np.append(z_diff, 0)
    # normalize the angle
    z_diff[1] = (z_diff[1] + math.pi) % (2 * math.pi) - math.pi

    mu_bar = mu_bar.reshape(3,1) + K_t @ z_diff.reshape(3,1)

    sigma_bar = (np.eye(3) - (K_t @ h_t)) @ sigma_bar

    return mu_bar.reshape(3), sigma_bar  

# @jit
# def sensor_reading_fast(mu_bar, sigma_bar, z, landmark):
    
#     q = (landmark[0] - mu_bar[0])**2 + (landmark[1] - mu_bar[1])**2
#     z_hat = jnp.array([jnp.sqrt(q), jnp.arctan2(landmark[1] - mu_bar[1], landmark[0] - mu_bar[0]) - mu_bar[2]-jnp.pi/2])
    
#     h_t = jnp.array([[-(landmark[0] - mu_bar[0])/jnp.sqrt(q), -(landmark[1] - mu_bar[1])/jnp.sqrt(q), 0],
#                     [(landmark[1] - mu_bar[1])/q, - (landmark[0] - mu_bar[0])/q, -1],
#                     [0, 0, 0]])
    
#     S_t = h_t @ sigma_bar @ h_t.T + jnp.array(Q_t)

#     K_t = jnp.array(sigma_bar @ h_t.T @ jnp.linalg.inv(S_t))  # Kalman Gain, inverting a matrix, will look to optimize this later
    
#     # z_hat[-1] -= math.pi/2
#     # z_hat = jax.ops.index_update(z_hat, -1, z_hat[-1] - jnp.pi/2)
#     z_diff = z[1:] - z_hat
#     # add a 0 to the end of the z_diff to account for the id
#     z_diff = jnp.append(z_diff, 0)
#     # normalize the angle
#     z_diff = z_diff.at[1].set((z_diff[1] + jnp.pi) % (2 * jnp.pi) - jnp.pi)

#     mu_bar = mu_bar.reshape(3,1) + K_t @ z_diff.reshape(3,1)

#     sigma_bar = (jnp.eye(3) - (K_t @ h_t)) @ sigma_bar

#     return mu_bar.reshape(3), sigma_bar 

def EKF(robot_odometry, initial_x, initial_y, initial_theta, robot_measurements, use_jit=False):    
    # We need to find when the robot measurements and odometry line up
    measurement_starting_idx = 0
    starting_timestep = robot_odometry[0,0]
    for i in range(robot_measurements.shape[0]):
        if robot_measurements[i+1,0] > starting_timestep:
            measurement_starting_idx = i
            break
    robot_measurements = robot_measurements[measurement_starting_idx:-1] # Robot measurements are now in sync with odometry data
    
    # x = initial_x # We need to know where the robot is starting from
    # y = initial_y
    # theta = initial_theta
    mu_bar = np.array([initial_x, initial_y, initial_theta])

    sigma_bar = np.array([[0.002, 0.002, 0.002],
                          [0.002, 0.002, 0.002],
                          [0.002, 0.002, 0.002]])

    positions = np.empty((robot1_odometry.shape[0], 3))
    # Add the initial position to the array
    positions[0] = mu_bar
    measurement_idx = 0

    time_stamp_sensor = [] #np.empty((robot1_odometry.shape[0], 4))
    sigma_bars = np.empty((robot1_odometry.shape[0], 3, 3))
    sigma_bars[0] = sigma_bar

    # Time to actually iterate through the data
    start_time = time.time()
    for i in range(1, robot_odometry.shape[0]):
        # Motion step
        dt = robot_odometry[i,0] - robot_odometry[i-1,0]
        mu_bar = motion_mu_bar(mu_bar, [robot_odometry[i][1], robot_odometry[i][2]], dt)
        sigma_bar = motion_sigma_bar(sigma_bar, mu_bar, [robot_odometry[i][1], robot_odometry[i][2]], dt, alphas)
        
        # Check to see if we have a sensor reading at this time step
        reading = None
        while(robot_odometry[i][0] >= robot_measurements[measurement_idx][0]):
            # Correction Step
            if False:# use_jit:
                pass
                # z = robot_measurements[measurement_idx][1:]
                # landmark_id = robot_association_dict[(z[0]).astype(int)]
                # landmark = landmark_dict[landmark_id]
                # mu_bar, sigma_bar = sensor_reading_fast(jnp.array(mu_bar), jnp.array(sigma_bar), robot_measurements[measurement_idx][1:], landmark)
                # # convert back to numpy
                # mu_bar = np.array(mu_bar)
                # sigma_bar = np.array(sigma_bar)
            else:
                mu_bar, sigma_bar = sensor_reading(mu_bar, sigma_bar, robot_measurements[measurement_idx][1:], landmark_dict)
            measurement_idx += 1
            reading = robot_measurements[measurement_idx][1:]
            if measurement_idx >= robot_measurements.shape[0]-5: # The end of the data set gets a little weird.
                print("Computation Time:", time.time() - start_time)
                return positions, sigma_bars, time_stamp_sensor

        positions[i] = mu_bar
        sigma_bars[i] = sigma_bar
        est = [robot_odometry[i][0], reading]
        time_stamp_sensor.append(est)
    return positions, sigma_bars, time_stamp_sensor
# sensor_reading(mu_bar, sigma_bar, z, Q, landmarks):
robot1_odometry = robot1_odometry[100:-1]
# find the starting GT position
gt_index = 0
while robot_1_gt[gt_index,0] < robot1_odometry[0,0]:
    gt_index += 1
print("Normal Speed")
ekf_positions, ekf_sigmas, ekf_time_stamp_sensors = EKF(robot1_odometry, robot_1_gt[gt_index,1], robot_1_gt[gt_index,2], robot_1_gt[gt_index,3], robot1_measurements)
print("JIT Speed")
ekf_positions_fast, ekf_sigmas_fast, ekf_time_stamp_sensors_fast = EKF(robot1_odometry, robot_1_gt[gt_index,1], robot_1_gt[gt_index,2], robot_1_gt[gt_index,3], robot1_measurements, use_jit=True)

# Plotting
# Static Plotting
# plot the ground truth landmarks
plt.figure()
index = 30000
gt_stop = gt_index
while robot_1_gt[gt_stop,0] < robot1_odometry[index,0]:
    gt_stop += 1

# plot circles around the landmarks
# for i in range(landmark_gt.shape[0]):
    # circle = plt.Circle((landmark_gt[i,1], landmark_gt[i,2]), 0.1, color='r', fill=False)
    # plt.gca().add_patch(circle)
plt.plot(landmark_gt[:,1], landmark_gt[:,2], 'r.', markersize=20)
# add labels to the landmarks
for i in range(landmark_gt.shape[0]):
    plt.text(landmark_gt[i,1], landmark_gt[i,2], str(int(landmark_gt[i,0])))

# plot the ground truth robot trajectory
plt.plot(robot_1_gt[gt_index:gt_stop,1], robot_1_gt[gt_index:gt_stop,2], 'b-', label='Ground Truth')
# plt.plot(robot_1_gt[gt_index:,1], robot_1_gt[gt_index:,2], 'b-', label='Ground Truth')

# plot dead reckoning
positions_dr = dead_reckoning(robot1_odometry, robot_1_gt[gt_index,1], robot_1_gt[gt_index,2], robot_1_gt[gt_index,3])
plt.plot(positions_dr[:index,0], positions_dr[:index,1], 'y-', label='Dead Reckoning')
# plt.plot(positions_dr[:,0], positions_dr[:,1], 'y-', label='Dead Reckoning')

# plot the EKF robot trajectory
plt.plot(ekf_positions[:index,0], ekf_positions[:index,1], 'g-', label='EKF')
# plt.plot(ekf_positions[:,0], ekf_positions[:,1], 'g-', label='EKF')

plt.legend()
plt.title('EKF')
plt.xlabel('x(m)')
plt.ylabel('y(m)')
# make the plot big!
plt.gcf().set_size_inches(12, 12)

# plot the MSE of the dead reckoning trajectory FIXME!
sse = []
prev_gt_idx = 0
for i in tqdm(range(index)):
    time_step = ekf_time_stamp_sensors[i][0]
    closest_gt_idx = prev_gt_idx
    while robot_1_gt[closest_gt_idx,0] < time_step:
        closest_gt_idx += 1
    sse.append((ekf_positions[i,0]- robot_1_gt[closest_gt_idx,1])**2 + (ekf_positions[i,1] - robot_1_gt[closest_gt_idx,2])**2)
    prev_gt_idx = closest_gt_idx

fig, ax2 = plt.subplots()

seg_to_time = int(robot1_odometry[-1][0] - robot1_odometry[0][0])
# get the MSE of the x and y positions
# sse = (ekf_positions[:seg,0]- robot_1_gt[gt_start:gt_start+seg,1])**2 + (ekf_positions[:seg,1] - robot_1_gt[gt_start:gt_start+seg,2])**2
ax2.plot(sse, 'r.', markersize=1)
# ax3.plot(mse[0], mse[1], 'r.', markersize=20)
ax2.grid()
ax2.set_title('Sum Squared Error of EKF after {} minutes'.format(int(seg_to_time/60)))
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Sum Squared Error(m)')
plt.show()

# Dynamic Plotting
if True:
    fig, ax = plt.subplots()
    measurement_flag = False
    for i in range(ekf_positions.shape[0]):
        if ekf_time_stamp_sensors[i][1] is not None:
            measurement_flag = True
        if measurement_flag:
            # If there's a measurement at this timestamp, plot it
            if measurement_flag:
                landmark_id = robot_association_dict[int(ekf_time_stamp_sensors[i][1][0])]
                landmark = landmark_dict[landmark_id]
                xmin = ekf_positions[i][0]
                xmax = landmark[0]
                ymin = ekf_positions[i][1]
                ymax = landmark[1]
                z_line = mlines.Line2D([xmin,xmax], [ymin,ymax], color='r', linewidth=2)
                ax.add_line(z_line)
                measurement = ekf_time_stamp_sensors[i][1]
                xmin = ekf_positions[i][0]
                xmax = ekf_positions[i][0] + measurement[1]*math.cos(ekf_positions[i][2] + measurement[2])
                ymin = ekf_positions[i][1]
                ymax = ekf_positions[i][1] + measurement[1]*math.sin(ekf_positions[i][2] + measurement[2])
                z_est = mlines.Line2D([xmin,xmax], [ymin,ymax], color='g', linewidth=2)
                ax.add_line(z_est)
            
            timestep = ekf_time_stamp_sensors[i][0]
            # find the ground truth position
            gt_index = 0
            while robot_1_gt[gt_index,0] < timestep:
                gt_index += 1
            # plot ground truth
            ax.scatter(robot_1_gt[:gt_index,1], robot_1_gt[:gt_index,2], color='blue', s=1)
            # plot robot positions
            ekf_positions[i,2] = (ekf_positions[i,2] + math.pi) % (2 * math.pi) - math.pi
            ax.scatter(ekf_positions[:i,0], ekf_positions[:i,1], color='black', s=1)
            for landmark in landmark_dict.values():
                ax.scatter(landmark[0], landmark[1], color='r')

            # Drawing robot and covariance ellipse
            cov_xy = ekf_sigmas[i][:2,:2] # covariance matrix for x and y only
            mean_xy = ekf_positions[i][:2]
            heading = ekf_positions[i][-1]
            eigvals, eigvecs = np.linalg.eig(cov_xy) # eigenvalues and eigenvectors of cov_xy
            theta = np.degrees(np.arctan2(*eigvecs[::-1, 0])) # angle of major axis (in degrees)
            # if e-values are negative, make them positive
            eigvals = np.abs(eigvals)
            width, height = 2 * np.sqrt(eigvals) # width and height of ellipse
            ellipse = Ellipse(xy=mean_xy, width=width, height=height, angle=theta, alpha=0.2)
            ax.add_patch(ellipse)

            # draw a little robot
            robot = plt.Circle((ekf_positions[i][0], ekf_positions[i][1]), .1, color='g', fill=True)
            # draw heading vector
            # plt.quiver(ekf_positions[i][0], ekf_positions[i][1], np.cos(heading), np.sin(heading), color='b', scale=10)
            ax.add_patch(robot)
            # add line for heading
            xmin = ekf_positions[i][0]
            xmax = ekf_positions[i][0]+0.2*(np.cos(heading))
            ymin = ekf_positions[i][1]
            ymax = ekf_positions[i][1]+0.2*(np.sin(heading))
            l = mlines.Line2D([xmin,xmax], [ymin,ymax], color='b', linewidth=2)
            ax.add_line(l)

            plt.xlim(-1,6)
            plt.ylim(-6,6)
            plt.show(block=False)
            if measurement_flag:
                plt.pause(interval=0.3)
            else:
                plt.pause(interval=0.1)
            measurement_flag = False
            ax.clear()