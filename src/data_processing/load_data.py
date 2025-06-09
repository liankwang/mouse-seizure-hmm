import pandas as pd
import numpy as np

from .downsample import downsample
from .kalman_imu import KalmanIMU
from src.utils import hourmin_to_timestamp


def load_data(path, start_time=None, end_time=None, pitch_roll=True, sampling_frequency=None):
    # Load data from a parquet file
    if path is None:
        raise ValueError("Path to data must be provided.")

    if path.suffix != '.parquet':
        raise ValueError("Please provide a .parquet file.")
    
    data = pd.read_parquet(path, engine='pyarrow')

    # Filter data by time range (if specified)
    if start_time is not None and end_time is not None:
        data = subset_data_by_time(data, start_time, end_time)
    else:
        print("No time range specified. Using full dataset. This may lead to memory issues for large datasets.")

    # Add roll and pitch features
    if pitch_roll:
        if sampling_frequency is None:
            raise ValueError("Sampling frequency must be provided for pitch and roll computation.")
        data['roll'], data['pitch'] = compute_roll_pitch(data, sampling_frequency) 

    return data


def process_data(df, 
                 downsampling_method=None, 
                 target_sps=None, original_sps=None, 
                 n_pca_components=10):
    feature_list = df.columns[2:].tolist()
    n_features = len(feature_list)
    print(f"Processing data with {n_features} features: {feature_list}")

    data = df[feature_list].values
    times = df['pi_time'].values

    # Downsample data
    if downsampling_method is None:
        print("No downsampling method specified. Using original data.")
    else:
        data, times = downsample(data, times, downsampling_method, target_sps, original_sps, n_pca_components)

    # Split the data into training and testing sets
    print("Splitting data into training and testing sets using 80/20 split...")
    split_idx = int(0.8 * len(data))
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    train_times = times[:split_idx]
    test_times = times[split_idx:]

    # Standardize the data
    print("Standardizing data...")
    train_data, mean, std = standardize_data(train_data)
    test_data = (test_data - mean) / std

    # Store data into dict
    train_dataset = [{'data': train_data,
                     'times': train_times}]
    test_dataset = [{'data': test_data,
                     'times': test_times}]
    
    return train_dataset, test_dataset


def standardize_data(data):
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    standardized_data = (data - mean) / std
    return standardized_data, mean, std


def compute_roll_pitch(data, sampling_frequency):
    print(f"Computing roll and pitch features with {sampling_frequency}Hz sampling frequency...")
    acc = data[['accX', 'accY', 'accZ']].values
    gyr = data[['gyroX', 'gyroY', 'gyroZ']].values

    # Apply Kalman filter
    kalman_filter = KalmanIMU(acc=acc, gyr=gyr, frequency=sampling_frequency)
    roll, pitch = kalman_filter.roll_pitch

    return roll, pitch
    

def subset_data_by_time(data, start_time, end_time):
    print(f"Keeping data in time range {start_time} to {end_time}.")
    start_timestamp = hourmin_to_timestamp(start_time)
    end_timestamp = hourmin_to_timestamp(end_time)
    return data[(data['pi_time'] >= start_timestamp) & (data['pi_time'] <= end_timestamp)]
    



