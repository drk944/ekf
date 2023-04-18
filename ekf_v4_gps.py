import numpy as np 
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Ellipse
import matplotlib.lines as mlines
from tqdm import tqdm


# robot association is an array where each row corresponds a bar code value to a landmark

# Landmark ground truth (where are the landmarks in the world)

# Robot 1 ground truth
robot_1_gt = np.loadtxt('datasets/MRCLAM_Dataset1/Robot1_Groundtruth.dat', skiprows=4, dtype='float')

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

sigma_x = 0.1
sigma_y = 0.1

Q_t = [[sigma_x**2, 0,],
       [0, sigma_y**2]]

def sensor_reading(mu_bar, sigma_bar, pos):

    q = (pos[0] - mu_bar[0])**2 + (pos - mu_bar[1])**2
    z_hat = np.array(pos[0]-mu_bar[0], pos[1]-mu_bar[1])
        
    h_t = np.array([[1, 0, 0],
                    [0, 1, 0]])
    
    S_t = h_t @ sigma_bar @ h_t.T + Q

    K_t = sigma_bar @ h_t.T @ np.linalg.inv(S_t)  # Kalman Gain, inverting a matrix, will look to optimize this later

    z_error = pos - mu_bar

    mu_bar = mu_bar.reshape(3,1) + K_t @ z_diff.reshape(3,1)
    # 3.32944, -3.152725, 2.5181
    sigma_bar = (np.eye(3) - (K_t @ h_t)) @ sigma_bar

    return mu_bar.reshape(3), sigma_bar  

alphas = [.04, .0009, .0081, .0064] # robot-dependent motion noise 
sigma_range = 2;
sigma_bearing = 3;
sigma_id = 1;

# robot-dependent sensor noise parameters
sigma_range = 2;
sigma_bearing = 3;
sigma_id = 1;

Q_t = np.diag([sigma_range^2, sigma_bearing^2, sigma_id^2])

def EKF(robot_odometry, initial_x, initial_y, initial_theta, robot_measurements):
    # We need to find when the robot measurements and odometry line up
    measurement_starting_idx = 0
    starting_timestep = robot_odometry[0,0]
    for i in range(robot_measurements.shape[0]):
        if robot_measurements[i,0] > starting_timestep:
            measurement_starting_idx = i
            break
    robot_measurements = robot_measurements[measurement_starting_idx:-1] # Robot measurements are now in sync with odometry data
    
    # x = initial_x # We need to know where the robot is starting from
    # y = initial_y
    # theta = initial_theta
    mu_bar = np.array([initial_x, initial_y, initial_theta])

    sigma_bar = np.array([[0.01, 0.01, 0.01],
                          [0.01, 0.01, 0.01],
                          [0.01, 0.01, 0.01]])

    positions = np.empty((robot1_odometry.shape[0], 3))
    pos_uncertainty = np.empty((robot1_odometry.shape[0], 3, 3))
    # Add the initial position to the array
    positions[0] = mu_bar
    pos_uncertainty[0] = sigma_bar
    measurement_idx = 0
    measurement_flag = False

    # Time to actually iterate through the data
    for i in tqdm(range(1, robot_odometry.shape[0])):
        # Motion step
        if i == 301:
            pass
        dt = robot_odometry[i,0] - robot_odometry[i-1,0]
        mu_bar = motion_mu_bar(mu_bar, [robot_odometry[i][1], robot_odometry[i][2]], dt)
        sigma_bar = motion_sigma_bar(sigma_bar, mu_bar, [robot_odometry[i][1], robot_odometry[i][2]], dt, alphas)
        
        # Check to see if we have a sensor reading at this time step
        while(robot_odometry[i][0] >= robot_measurements[measurement_idx+1][0]):
            # Correction Step
            measurement_flag = True
            measurement_idx += 1
            mu_bar, sigma_bar = sensor_reading(mu_bar, sigma_bar, robot_measurements[measurement_idx][1:], Q_t, landmark_dict)
 
            if measurement_idx >= robot_measurements.shape[0]-5: # The end of the data set gets a little weird.
                return positions, pos_uncertainty

        positions[i] = mu_bar
        pos_uncertainty[i] = sigma_bar

    return positions, pos_uncertainty

# sensor_reading(mu_bar, sigma_bar, z, Q, landmarks):
robot1_odometry = robot1_odometry[100:-1]
# find the starting GT position
gt_index = 0
while robot_1_gt[gt_index,0] < robot1_odometry[0,0]:
    gt_index += 1
meas_idx = 0
while robot1_measurements[meas_idx,0] < robot1_odometry[0,0]:
    meas_idx += 1
robot1_measurements = robot1_measurements[meas_idx:-1]

ekf_positions, ekf_pos_uncertainty = EKF(robot1_odometry, robot_1_gt[gt_index,1], robot_1_gt[gt_index,2], robot_1_gt[gt_index,3], robot1_measurements)

# plot the ground truth landmarks
plt.figure()
index = 5000
# plot circles around the landmarks
# for i in range(landmark_gt.shape[0]):
    # circle = plt.Circle((landmark_gt[i,1], landmark_gt[i,2]), 0.1, color='r', fill=False)
    # plt.gca().add_patch(circle)
plt.plot(landmark_gt[:,1], landmark_gt[:,2], 'r.', markersize=20)
# add labels to the landmarks
for i in range(landmark_gt.shape[0]):
    plt.text(landmark_gt[i,1], landmark_gt[i,2], str(int(landmark_gt[i,0])))

# plot the ground truth robot trajectory
plt.plot(robot_1_gt[gt_index:gt_index+index,1], robot_1_gt[gt_index:gt_index+index,2], 'b-', label='Ground Truth')
# plt.plot(robot_1_gt[gt_index:,1], robot_1_gt[gt_index:,2], 'b-', label='Ground Truth')

# plot dead reckoning
positions_dr = dead_reckoning(robot1_odometry, robot_1_gt[gt_index,1], robot_1_gt[gt_index,2], robot_1_gt[gt_index,3])
plt.plot(positions_dr[:index,0], positions_dr[:index,1], 'y-', label='Dead Reckoning')
# plt.plot(positions_dr[:,0], positions_dr[:,1], 'y-', label='Dead Reckoning')

# plot the EKF robot trajectory
plt.plot(ekf_positions[:index,0], ekf_positions[:index,1], 'g-', label='EKF')
# plt.plot(ekf_positions[:,0], ekf_positions[:,1], 'g-', label='EKF')

# Generate an animation to show the first 100 timesteps


plt.legend()
plt.title('EKF')
plt.xlabel('x(m)')
plt.ylabel('y(m)')
# make the plot big!
plt.gcf().set_size_inches(12, 12)
# plt.show()


plt.xlim(-1, 6)
plt.ylim(-6, 6)
plt.show()