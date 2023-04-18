import numpy as np 
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Ellipse
import matplotlib.lines as mlines


# robot association is an array where each row corresponds a bar code value to a landmark
robot_association = np.loadtxt('datasets/MRCLAM_Dataset1/Barcodes.dat', skiprows=4, dtype='int')
# create empy dictionary
robot_association_dict = {}
# iterate through each row
for row in robot_association:
    # add the key value pair to the dictionary
    if row[0] == 2 or row[0] == 3 or row[0] == 4 or row[0] == 5:
        continue # ignore the other robots in the dataset
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
robot1_measurements = robot1_measurements[robot1_measurements[:,1] != 5]
robot1_measurements = robot1_measurements[robot1_measurements[:,1] != 14]
robot1_measurements = robot1_measurements[robot1_measurements[:,1] != 41]
robot1_measurements = robot1_measurements[robot1_measurements[:,1] != 32]
robot1_measurements = robot1_measurements[robot1_measurements[:,1] != 23]
# robot1_measurements = robot1_measurements[robot1_measurements[:,1] != 18] # Marker 18 is dumb!

# remove all rows with duplicate timesteps
# robot1_measurements = robot1_measurements[np.unique(robot1_measurements[:,0], return_index=True)[1]]

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

sigma_range = 2
sigma_bearing = 3
sigma_id = 1
Q_t = [[sigma_range**2, 0, 0,],
       [0, sigma_bearing**2, 0],
       [0, 0, sigma_id**2]]

def sensor_reading(mu_bar, sigma_bar, z, Q, landmarks):
    try:
        landmark_id = robot_association_dict[int(z[0])]
        landmark = landmark_dict[landmark_id]
    except KeyError:
        print("ERROR!") # This should never happen because I cleaned the data
        return mu_bar, sigma_bar
    q = (landmark[0] - mu_bar[0])**2 + (landmark[1] - mu_bar[1])**2
    z_hat = np.array([math.sqrt(q), math.atan2(landmark[1] - mu_bar[1], landmark[0] - mu_bar[0]) - mu_bar[2]])
    z_hat[1] = (z_hat[1] + np.pi) % (2 * np.pi) - np.pi

    h_t = np.array([[-(landmark[0] - mu_bar[0])/math.sqrt(q), -(landmark[1] - mu_bar[1])/math.sqrt(q), 0],
                    [(landmark[1] - mu_bar[1])/q, - (landmark[0] - mu_bar[0])/q, -1],
                    [0, 0, 0]])
    
    S_t = h_t @ sigma_bar @ h_t.T + Q

    K_t = sigma_bar @ h_t.T @ np.linalg.inv(S_t)  # Kalman Gain, inverting a matrix, will look to optimize this later

    z_diff = z[1:] - z_hat
    # add a 0 to the end of the z_diff to account for the id
    z_diff = np.append(z_diff, 0)
    # normalize the bearing to be within -pi and pi
    # z_diff[1] = (z_diff[1] + math.pi) % (2 * math.pi) - math.pi

    mu_bar = mu_bar.reshape(3,1) + K_t @ z_diff.reshape(3,1)
    # 3.32944, -3.152725, 2.5181
    sigma_bar = (np.eye(3) - (K_t @ h_t)) @ sigma_bar

    return mu_bar.reshape(3), sigma_bar  

alphas = [.04, .0009, .0081, .0064] # robot-dependent motion noise 
sigma_range = 2;
sigma_bearing = 3;
sigma_id = 1;

# robot-dependent sensor noise parameters
sigma_range = 4;
sigma_bearing = 9;
sigma_id = 1;

Q_t = np.diag([sigma_range^2, sigma_bearing^2, sigma_id^2])

def EKF(robot_odometry, initial_x, initial_y, initial_theta, robot_measurements):
    fig, ax = plt.subplots()
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
    for i in range(1, robot_odometry.shape[0]):
        print(i)
        if i == 301:
            pass
        ax.scatter(robot_1_gt[:5000,1], robot_1_gt[:5000,2], color='y', s=1)
        ax.scatter(positions[:,0], positions[:,1], color='black', s=1)
        for landmark in landmark_dict.values():
            ax.scatter(landmark[0], landmark[1], color='r')
        # Motion step
        dt = robot_odometry[i,0] - robot_odometry[i-1,0]
        mu_bar = motion_mu_bar(mu_bar, [robot_odometry[i][1], robot_odometry[i][2]], dt)
        sigma_bar = motion_sigma_bar(sigma_bar, mu_bar, [robot_odometry[i][1], robot_odometry[i][2]], dt, alphas)
        
        # Check to see if we have a sensor reading at this time step
        while(robot_odometry[i][0] >= robot_measurements[measurement_idx+1][0]):
            # Correction Step
            print("Meas IDX:", measurement_idx+1)
            measurement_flag = True
            measurement_idx += 1
            mu_bar, sigma_bar = sensor_reading(mu_bar, sigma_bar, robot_measurements[measurement_idx][1:], Q_t, landmark_dict)
            try:
                landmark_id = robot_association_dict[int(robot_measurements[measurement_idx][1])]
                landmark = landmark_dict[landmark_id]
                xmin = mu_bar[0]
                xmax = landmark[0]
                ymin = mu_bar[1]
                ymax = landmark[1]
                z_line = mlines.Line2D([xmin,xmax], [ymin,ymax], color='r', linewidth=2)
                ax.add_line(z_line)
            except:
                print("ERROR: Should not reach this")
                pass

            xmin = mu_bar[0]
            xmax = mu_bar[0] + robot_measurements[measurement_idx][2]*math.cos(mu_bar[2] + robot_measurements[measurement_idx][3])
            ymin = mu_bar[1]
            ymax = mu_bar[1] + robot_measurements[measurement_idx][2]*math.sin(mu_bar[2] + robot_measurements[measurement_idx][3])
            z_est = mlines.Line2D([xmin,xmax], [ymin,ymax], color='g', linewidth=2)
            ax.add_line(z_est)
            if measurement_idx >= robot_measurements.shape[0]-5: # The end of the data set gets a little weird.
                return positions, pos_uncertainty

        positions[i] = mu_bar
        pos_uncertainty[i] = sigma_bar
        if i % 10 == 0 or measurement_flag:
            cov_xy = pos_uncertainty[i][:2,:2] # covariance matrix for x and y only
            mean_xy = positions[i][:2]
            heading = positions[i][-1]
            eigvals, eigvecs = np.linalg.eig(cov_xy) # eigenvalues and eigenvectors of cov_xy
            theta = np.degrees(np.arctan2(*eigvecs[::-1, 0])) # angle of major axis (in degrees)
            # if e-values are negative, make them positive
            eigvals = np.abs(eigvals)
            width, height = 2 * np.sqrt(eigvals) # width and height of ellipse
            ellipse = Ellipse(xy=mean_xy, width=width, height=height, angle=theta, alpha=0.2)
            ax.add_patch(ellipse)

            # draw a little robot
            robot = plt.Circle((positions[i][0], positions[i][1]), .1, color='g', fill=False)
            # draw heading vector
            # plt.quiver(ekf_positions[i][0], ekf_positions[i][1], np.cos(heading), np.sin(heading), color='b', scale=10)
            ax.add_patch(robot)
            # add line for heading
            xmin = positions[i][0]
            xmax = positions[i][0]+0.2*(np.cos(heading))
            ymin = positions[i][1]
            ymax = positions[i][1]+0.2*(np.sin(heading))
            l = mlines.Line2D([xmin,xmax], [ymin,ymax], color='b', linewidth=2)
            ax.add_line(l)
            plt.xlim(0,6)
            plt.ylim(-6,6)
            plt.show(block=False)
            # if i > 295:
            #     plt.pause(interval=1)
            if measurement_flag:
                plt.pause(interval=1)
            else:
                plt.pause(interval=0.1)
            ax.clear()
            measurement_flag = False
    # print size of positions
    print(positions.shape)
    print(pos_uncertainty.shape)
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
################################################################
x = ekf_positions[:, 0]
y = ekf_positions[:, 1]

# Create figure and axes objects

# Scatter plot of positions
# ax.scatter(x, y)

for i in range(100): #ekf_positions.shape[0]):
    # plot landmarks
    if i % 5 == 0:
        continue
    for landmark in landmark_dict.values():
        ax.scatter(landmark[0], landmark[1], color='r')
# Compute and plot confidence ellipses
    cov_xy = ekf_pos_uncertainty[i][:2,:2] # covariance matrix for x and y only
    mean_xy = ekf_positions[i][:2]
    heading = ekf_positions[i][-1]
    eigvals, eigvecs = np.linalg.eig(cov_xy) # eigenvalues and eigenvectors of cov_xy
    theta = np.degrees(np.arctan2(*eigvecs[::-1, 0])) # angle of major axis (in degrees)
    width, height = 2 * np.sqrt(eigvals) # width and height of ellipse
    ellipse = Ellipse(xy=mean_xy, width=width, height=height, angle=theta, alpha=0.2)
    ax.add_patch(ellipse)

    # draw a little robot
    robot = plt.Circle((ekf_positions[i][0], ekf_positions[i][1]), .05, color='g', fill=False)
    # draw heading vector
    # plt.quiver(ekf_positions[i][0], ekf_positions[i][1], np.cos(heading), np.sin(heading), color='b', scale=10)
    ax.add_patch(robot)
    # add line for heading
    xmin = ekf_positions[i][0]
    xmax = ekf_positions[i][0]+0.1*(np.cos(heading))
    ymin = ekf_positions[i][1]
    ymax = ekf_positions[i][1]+0.1*(np.sin(heading))
    l = mlines.Line2D([xmin,xmax], [ymin,ymax], color='b', linewidth=2)
    ax.add_line(l)
    # plt.Line2D((ekf_positions[i][0], ekf_positions[i][0]+np.cos(heading)), (ekf_positions[i][1], ekf_positions[i][1]+np.sin(heading)), color='b')
    plt.xlim(1.5,3.5)
    plt.ylim(-4,-2)
    # plt.axis('equal')
    plt.show(block=False)
    plt.pause(interval=0.1)
    ax.clear()


# Show plot
plt.show()
################################################################
# plot the ground truth landmarks
plt.figure()
index = 1000
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