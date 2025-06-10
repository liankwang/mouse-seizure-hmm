import pandas as pd
import numpy as np

from .downsample import downsample
from .kalman_imu import KalmanIMU
from src.utils import hourmin_to_timestamp_ms

def load_data(path, 
              seizure_path,
              video_start_pi_time_sec, 
              start_time=None, end_time=None, 
              pitch_roll=True, 
              sampling_frequency=None):
    # Load data from a parquet file
    if path is None:
        raise ValueError("Path to data must be provided.")

    if path.suffix != '.parquet':
        raise ValueError("Please provide a .parquet file.")
    
    # Add seizure labels if not already
    if not path.name.endswith('_with_seizures.parquet'):
        if seizure_path is None:
            raise ValueError("Please provide a path to seizure labels.")
        
        print("Adding seizure labels into data...")
        data = pd.read_parquet(path, engine='pyarrow')
        seizures = read_seizures(seizure_path, video_start_pi_time_sec)
        data['seizure_status'] = assign_labels(data, seizures)
    else:
        data = pd.read_parquet(path, engine='pyarrow')

    # Filter data by time range (if specified)
    if start_time is not None and end_time is not None:
        data = subset_data_by_time(data, start_time, end_time, video_start_pi_time_sec)
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

    # Downsample data
    if downsampling_method is None:
        print("No downsampling method specified. Using original data.")
    else:
        data, times, seizures = downsample(df, downsampling_method, target_sps, original_sps, n_pca_components=n_pca_components)

    # Split the data into training and testing sets
    print("Splitting data into training and testing sets using 80/20 split...")
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    train_times = times[:split_idx]
    test_times = times[split_idx:]
    train_seizures = seizures[:split_idx]
    test_seizures = seizures[split_idx:]

    # Standardize the data
    print("Standardizing data...")
    train_data, mean, std = standardize_data(train_data)
    test_data = (test_data - mean) / std

    # Store data into dict
    train_dataset = [{'data': train_data,
                     'times': train_times,
                     'seizures': train_seizures}]
    test_dataset = [{'data': test_data,
                     'times': test_times,
                     'seizures': test_seizures}]
    
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
    

def subset_data_by_time(data, start_time, end_time, video_start_pi_time_sec):
    print(f"Keeping data in time range {start_time} to {end_time}.")
    if video_start_pi_time_sec is None:
        raise ValueError("video_start_pi_time_sec must be provided to subset data by time.")
    start_timestamp = hourmin_to_timestamp_ms(start_time) + video_start_pi_time_sec * 1000
    end_timestamp = hourmin_to_timestamp_ms(end_time) + video_start_pi_time_sec * 1000
    return data[(data['pi_time'] >= start_timestamp) & (data['pi_time'] <= end_timestamp)]
    


def assign_labels(data, labels):
    intervals = pd.IntervalIndex.from_arrays(
        labels['start_pi_time'],
        labels['end_pi_time'],
        closed='left'
    )
    interval_locs = intervals.get_indexer(data['pi_time'])

    labeled_data = np.full(len(data), fill_value=np.nan, dtype=object)
    valid_locs = interval_locs != -1
    labeled_data[valid_locs] = labels.iloc[interval_locs[valid_locs]]['actual_status'].values
    return labeled_data

def read_seizures(path, video_start_pi_time_sec):
    seizures = pd.read_parquet(path, engine='pyarrow')
    seizures['start_pi_time'] = seizures['StartTime'] + video_start_pi_time_sec * 1000
    seizures['end_pi_time'] = seizures['start_pi_time'].shift(-1)
    seizures['end_pi_time'].iloc[-1] = seizures['start_pi_time'].iloc[-1] + 20.0 * 1000  # Add 20 ms to the last seizure end time
    return seizures