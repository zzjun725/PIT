import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_csv(file_path, columns_to_load):
    """
    Args:
    file_path: Path to the CSV file.
    columns_to_load: List of column names to load.

    Returns:
    data_series: A dictionary with the specified columns from the CSV file.
    """
    data_series = {}
    for column in columns_to_load:
        clean_series = pd.read_csv(file_path, usecols=['__time', column], sep=',').dropna()
        time_offset = clean_series['__time'].iloc[0]
        clean_series['__time'] = clean_series['__time'] - time_offset
        data_series[column] = clean_series

    return data_series


def interpolate_columns(data_series, time_series, columns):
    interpolated_states = {}
    # input_time = data_series[input_columns[0]]['__time']

    for column in columns:
        state_series = data_series[column]
        interpolated_values = np.interp(time_series, state_series['__time'], state_series[column])
        interpolated_states[column] = interpolated_values

    aligned_data = pd.DataFrame(interpolated_states, time_series)

    return aligned_data


def fetch_aligned_trajectory_data(file_path, sample_interval=0.05):
    columns_to_load = ['/ackermann_cmd/drive/speed', '/ackermann_cmd/drive/steering_angle',
                       '/odom/twist/twist/linear/x', '/pf/pose/odom/pose/pose/position/x',
                       '/pf/pose/odom/pose/pose/position/y', '/pf/pose/odom/pose/pose/orientation/yaw_deg']
    control_columns = ['/ackermann_cmd/drive/speed', '/ackermann_cmd/drive/steering_angle']
    state_columns = ['/odom/twist/twist/linear/x', '/pf/pose/odom/pose/pose/position/x', '/pf/pose/odom/pose/pose/position/y',
                     '/pf/pose/odom/pose/pose/orientation/yaw_deg']
    data_series = load_csv(file_path, columns_to_load)
    control_time_series = data_series[control_columns[0]]['__time']
    start_time = control_time_series.min()
    end_time = control_time_series.max()
    evenly_timestamps = np.arange(start_time, end_time+sample_interval, sample_interval)
    aligned_data = interpolate_columns(data_series, evenly_timestamps, columns_to_load)
    # aligned_data = interpolate_columns(data_series, data_series[control_columns[0]]['__time'], columns_to_load)
    # Cut from the first non-zero speed value
    first_non_zero_index = aligned_data[aligned_data['/ackermann_cmd/drive/speed'] > 0].index.min()
    aligned_data_cut = aligned_data.loc[first_non_zero_index:]

    return aligned_data_cut


if __name__ == "__main__":
    file_path = "/home/zzjun/Projects/PIT/datasets/levine_1.csv"
    aligned_data_cut = fetch_aligned_trajectory_data(file_path)

    # Setup for the subplots
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Plot for speed and linear velocity x
    # ax[0].plot(aligned_data_cut.index, aligned_data_cut['/ackermann_cmd/drive/speed'], label='Speed', color='blue')
    # ax[0].plot(aligned_data_cut.index, aligned_data_cut['/odom/twist/twist/linear/x'], label='Linear Velocity X',
    #            color='red')
    # ax[0].set_xlabel('Time')
    # ax[0].set_ylabel('Value')
    # ax[0].set_title('Speed and Linear Velocity X Over Time')
    ax[0].plot(aligned_data_cut.index, aligned_data_cut['/pf/pose/odom/pose/pose/orientation/yaw_deg'], label='Yaw', color='blue')
    ax[0].legend()
    ax[0].grid(True)

    # Plot for XY trajectory
    ax[1].plot(aligned_data_cut['/pf/pose/odom/pose/pose/position/x'],
               aligned_data_cut['/pf/pose/odom/pose/pose/position/y'], label='Trajectory', color='green')
    ax[1].set_xlabel('/pf/pose/odom/pose/pose/position/x')
    ax[1].set_ylabel('/pf/pose/odom/pose/pose/position/y')
    ax[1].set_title('XY Trajectory')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()