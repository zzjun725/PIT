from pit.integration import Euler, RK4
from pit.data_processor.csv_loader import fetch_aligned_trajectory_data, load_csv
import matplotlib.pyplot as plt


def moving_average(data_frame, window_size):
    """
    Compute the moving average of the DataFrame.

    Args:
        data_frame (pd.DataFrame): DataFrame with the data.
        window_size (int): Number of samples over which to compute the moving average.

    Returns:
        pd.DataFrame: DataFrame with the moving average added as a new column.
    """
    return data_frame.rolling(window=window_size).mean()


if __name__ == "__main__":
    file_path = "/home/zzjun/Projects/PIT/datasets/skir_4_1.csv"
    columns_to_load = ["/sensors/imu/imu/linear_acceleration/x", "/sensors/imu/imu/linear_acceleration/y"]
    data_series = load_csv(file_path, columns_to_load)


    # Compute the moving average for each column
    window_size = 100  # Example window size
    for column, df in data_series.items():
        df['moving_avg'] = moving_average(df[column], window_size)

    # Plotting
    plt.figure(figsize=(10, 6))
    for column, df in data_series.items():
        plt.plot(df['__time'], df['moving_avg'], label=f'Moving Avg of {column}')

    plt.title('Moving Average of IMU Linear Acceleration X')
    plt.xlabel('Time (offset in seconds)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.legend()
    plt.grid(True)
    plt.show()