from f1tenth_rl_obs.env_adapters.dynamics.bicycle_f110 import VehicleDynamicModel
import numpy as np
import matplotlib.pyplot as plt

"""
C_Pf:
- 3.12
- 2.23
- 0.72
- 0.23
C_Pr:
- 29.91
- 2.23
- 1.21
- 0.92
I_z: 0.09
a_max: 3
a_min: -3
h_cg: 0.02
l_f: 0.162
l_r: 0.145
l_wb: 0.307
m: 3.31
model_name: NUC4
mu: 1
tau_steer: 0.15
tire_model: pacejka

B_f = p.C_Pf[0]
C_f = p.C_Pf[1]
D_f = p.C_Pf[2]
E_f = p.C_Pf[3]
B_r = p.C_Pr[0]
C_r = p.C_Pr[1]
D_r = p.C_Pr[2]
E_r = p.C_Pr[3]
"""

np.random.seed(0)


def sample_trajectory_dynamic_st():
    # Vehicle parameters
    params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074,
              'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2,
              'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min': -5.0, 'v_max': 11.0}
    vehicle_model = VehicleDynamicModel('dynamic_ST', params)

    # Initial state [x1, x2, x3, x4:v, x5, x6, x7]
    s0 = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])  # Start with simple initial conditions

    # Define a sequence of controls [steer velocity, longitudinal acceleration]
    t = 0.5  # total time
    dt = 0.05  # time step
    n = int(t / dt)  # number of time steps
    # u_list = np.array([[(np.random.random()-0.2)*0.1, (np.random.random()-0.2)*3] for _ in range(n)])
    u_list = np.array([[0.0, 6.0] for _ in range(n)])

    plot_features = ['steering_angles', 'velocities']
    # plot_features = ['steering_angles']
    features = {}

    # Generate trajectory
    trajectory = vehicle_model.forward_trajectory(s0, u_list, dt)

    # Extract data for plots
    x = [state[0] for state in trajectory]
    y = [state[1] for state in trajectory]
    steering_angles = [state[2] for state in trajectory]
    features['steering_angles'] = steering_angles
    velocities = [state[3] for state in trajectory]
    features['velocities'] = velocities
    yaws = [state[4] for state in trajectory]
    features['yaws'] = yaws

    # Create plots
    fig, axs = plt.subplots(len(plot_features) + 1, 1, figsize=(10, 15))

    # Plot trajectory
    quiver = axs[0].quiver(x, y, np.cos(yaws), np.sin(yaws), velocities, scale=20, cmap='coolwarm')
    scatter = axs[0].scatter(x, y, c=velocities, cmap='coolwarm', label='Velocity')
    axs[0].set_xlabel('X Position (m)')
    axs[0].set_ylabel('Y Position (m)')
    axs[0].set_title('Vehicle Trajectory')
    axs[0].axis('equal')
    fig.colorbar(scatter, ax=axs[0], label='Velocity (m/s)')

    # Plot steering angles
    ax_idx = 1
    for feature_name in plot_features:
        axs[ax_idx].plot(features[feature_name], 'o-')
        axs[ax_idx].set_xlabel('Time step')
        axs[ax_idx].set_ylabel(f'{feature_name}')
        axs[ax_idx].set_title(f'{feature_name} Over Time')
        axs[ax_idx].grid(True)
        ax_idx += 1

    plt.tight_layout()
    plt.show()


def sample_trajectory_st_steer(tire_model='linear'):
    # a_max" 9.51
    if tire_model == 'linear':
        params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074,
                  'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2,
                  'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'a_min': -6.0, 'v_min': -5.0, 'v_max': 11.0,
                  'steer_delay_time': 0.1, "accel_delay_time": 0.05, "tire_model": "linear"}
    elif tire_model == 'pacejka':
        params = {
            "I": 0.09, "a_max": 5.0, "a_min": -3.0, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319,
            "h": 0.02, "lf": 0.162, "lr": 0.145, "lwb": 0.307, "m": 3.31,
            "mu": 1.0, "tau_steer": 0.15, "B_f": 3.12, "C_f": 2.23, "D_f": 0.72, "E_f": 0.23, "B_r": 29.91, "C_r": 2.23, "D_r": 1.21, "E_r": 0.92,
            'steer_delay_time': 0.05, "accel_delay_time": 0.05, "tire_model": "pacejka"
        }

    # vehicle_model = VehicleDynamicModel('kinematic_st_steer', params)
    vehicle_model = VehicleDynamicModel('dynamic_st_v1', params)
    # Initial state: [x0, x1, x2, x3, x4, x5, x6]
    #  x0: x-position in a global coordinate system
    #  x1: y-position in a global coordinate system
    #  x2: steering angle of front wheels
    #  x3: yaw angle
    #  x4: velocity in x-direction
    #  x5: velocity in y direction
    #  x6: yaw rate

    s0 = np.array([0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0])  # Start with simple initial conditions

    # Define a sequence of controls [steer, speed]
    t = 2.0  # total time
    dt = 0.04  # time step
    n = int(t / dt)  # number of time steps
    u_list = np.array([[0.3, 5.0] for _ in range(n)])
    # for i in range(1, n, 2):
    #     u_list[i] = [-0.3, 3.0]

    # plot_features = ['steering_angles', 'velocities', 'yaws']
    plot_features = ['yaw_rate', 'vy']
    features = {}

    # Generate trajectory
    trajectory = vehicle_model.forward_trajectory(s0, u_list, dt)

    # Extract data for plots
    x = [state[0] for state in trajectory]
    y = [state[1] for state in trajectory]
    steering_angles = [state[2] for state in trajectory]
    features['steering_angles'] = steering_angles
    velocities = [state[4] for state in trajectory]
    features['velocities'] = velocities
    vy = [state[5] for state in trajectory]
    features['vy'] = vy
    yaws = [state[3] for state in trajectory]
    features['yaw'] = yaws
    yaw_rate = [state[6] for state in trajectory]
    features['yaw_rate'] = yaw_rate

    # Create plots
    fig, axs = plt.subplots(len(plot_features) + 1, 1, figsize=(10, 15))

    # Plot trajectory
    quiver = axs[0].quiver(x, y, np.cos(yaws), np.sin(yaws), velocities, scale=20, cmap='coolwarm')
    scatter = axs[0].scatter(x, y, c=velocities, cmap='coolwarm', label='Velocity')
    axs[0].set_xlabel('X Position (m)')
    axs[0].set_ylabel('Y Position (m)')
    axs[0].set_title('Vehicle Trajectory')
    axs[0].axis('equal')
    fig.colorbar(scatter, ax=axs[0], label='Velocity (m/s)')

    # Plot steering angles
    ax_idx = 1
    for feature_name in plot_features:
        axs[ax_idx].plot(features[feature_name], 'o-')
        axs[ax_idx].set_xlabel('Time step')
        axs[ax_idx].set_ylabel(f'{feature_name}')
        axs[ax_idx].set_title(f'{feature_name} Over Time')
        axs[ax_idx].grid(True)
        ax_idx += 1

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sample_trajectory_st_steer(tire_model='pacejka')
