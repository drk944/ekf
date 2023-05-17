# Extended Kalman Filter (EKF) Localization Project

This project implements an Extended Kalman Filter (EKF) algorithm for mobile robot localization using range and bearing measurements to landmarks. The dataset used for this project is from Tim Barfoot's group, which includes measurements collected from a mobile robot moving in a known environment.

## Dataset

The dataset used in this project is provided by Tim Barfoot's group and consists of measurements collected from a mobile robot. It includes range and bearing measurements to landmarks in the environment, as well as odometry information in the format of forward and angular velocity. The dataset serves as the input for the EKF algorithm to estimate the robot's pose (position and orientation) in the environment.

## EKF Algorithm

The Extended Kalman Filter (EKF) is a recursive state estimation algorithm that combines the principles of the Kalman Filter with non-linear motion and measurement models. In this project, the EKF algorithm is applied to estimate the robot's pose based on the received range and bearing measurements to landmarks.

The EKF algorithm consists of two main steps:

1. **Prediction Step**: Using the odometry information, the algorithm predicts the next state (pose) of the robot by updating the mean and covariance of the estimated state. This step accounts for the motion dynamics of the robot.

2. **Update Step**: In this step, the algorithm incorporates the range and bearing measurements to landmarks to refine the estimated state. The predicted state is compared with the actual measurements, and the estimated state is updated using the EKF equations. This step improves the accuracy of the pose estimation.

## Usage

The Jupyter notebook goes through each aspect of the EKF algorithm with the math behind each step. ekf.py runs the entire algorithm and ekf_animation and ekf_animation_2 provide 2 different animated visualizations of the robot

## Acknowledgments

- The dataset used in this project is provided by Tim Barfoot's group at http://asrl.utias.utoronto.ca/datasets/mrclam/

- The EKF algorithm implementation is based on the textbook Probabilistic Robotics by Sebastian Thrun, Wolfram Burgard, and Dieter Fox.

- The modified motion model was borrowed from Andrew J Kramer. http://andrewjkramer.net/intro-to-the-ekf-step-1/

## License


