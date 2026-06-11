import pandas as pd
import numpy as np
import json

def calculate_duration(df):
    """
    Calculates the duration of a trip in minutes.
    Assures negative durations are set to 0.
    """
    df = df.copy()
    df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']) / np.timedelta64(1, 'm')
    df.loc[df['duration'] < 0, 'duration'] = 0.0
    return df

def compute_route_stats(df_train):
    """
    Computes average distance and duration for each pickup-dropoff route
    from the training set. Returns a dict containing the maps and global averages
    to serve as a fallback for unseen routes.
    """
    df_train = df_train.copy()
    df_train = calculate_duration(df_train)
    
    # Create route column
    df_train['pickup_dropoff'] = df_train['PULocationID'].astype(str) + ' ' + df_train['DOLocationID'].astype(str)
    
    # Compute group averages
    grouped_dist = df_train.groupby('pickup_dropoff')['trip_distance'].mean().to_dict()
    grouped_dur = df_train.groupby('pickup_dropoff')['duration'].mean().to_dict()
    
    # Compute global averages
    global_avg_dist = float(df_train['trip_distance'].mean())
    global_avg_dur = float(df_train['duration'].mean())
    
    stats = {
        "grouped_dist": grouped_dist,
        "grouped_dur": grouped_dur,
        "global_avg_dist": global_avg_dist,
        "global_avg_dur": global_avg_dur
    }
    return stats

def apply_route_stats(df, stats):
    """
    Applies the pre-calculated route statistics to a dataset.
    Fills unseen routes with global training averages to prevent NaNs.
    """
    df = df.copy()
    
    # Ensure duration exists
    if 'duration' not in df.columns:
        df = calculate_duration(df)
        
    df['pickup_dropoff'] = df['PULocationID'].astype(str) + ' ' + df['DOLocationID'].astype(str)
    
    # Map averages
    df['mean_distance'] = df['pickup_dropoff'].map(stats['grouped_dist'])
    df['mean_duration'] = df['pickup_dropoff'].map(stats['grouped_dur'])
    
    # Fill unseen routes with global average
    df['mean_distance'] = df['mean_distance'].fillna(stats['global_avg_dist'])
    df['mean_duration'] = df['mean_duration'].fillna(stats['global_avg_dur'])
    
    return df

def create_time_features(df):
    """
    Extracts time of day, day of week, and rush hour features.
    """
    df = df.copy()
    
    # Day and Month string categories
    df['day'] = df['tpep_pickup_datetime'].dt.day_name().str.lower()
    df['month'] = df['tpep_pickup_datetime'].dt.strftime('%b').str.lower()
    
    # Rush Hour flag (Mon-Fri, 6-10 and 16-20)
    hour = df['tpep_pickup_datetime'].dt.hour
    is_weekday = ~df['day'].isin(['saturday', 'sunday'])
    is_rush_time = ((hour >= 6) & (hour < 10)) | ((hour >= 16) & (hour < 20))
    df['rush_hour'] = (is_weekday & is_rush_time).astype(int)
    
    # Time-of-day bins for classification
    df['am_rush'] = ((hour >= 6) & (hour < 10)).astype(int)
    df['daytime'] = ((hour >= 10) & (hour < 16)).astype(int)
    df['pm_rush'] = ((hour >= 16) & (hour < 20)).astype(int)
    
    # Nighttime bin [20:00 - 06:00)
    df['nighttime'] = ((hour >= 20) | (hour < 6)).astype(int)
    
    return df

def save_route_stats(stats, filepath):
    with open(filepath, "w") as f:
        json.dump(stats, f, indent=4)

def load_route_stats(filepath):
    with open(filepath, "r") as f:
        return json.load(f)
