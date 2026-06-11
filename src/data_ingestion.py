import os
import urllib.request
import ssl
import pandas as pd
import numpy as np
import yaml

# Bypass SSL verification to download dataset on macOS python environments
ssl._create_default_https_context = ssl._create_unverified_context

def load_config(config_path=None):
    if config_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "config", "config.yaml")
    else:
        # If config_path is given, project_root is parent of its parent dir (since config is in config/)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Resolve relative paths inside the project structure
    for key in ["raw_data_dir", "processed_data_dir", "models_dir"]:
        config["paths"][key] = os.path.join(project_root, config["paths"][key])
        
    return config

def download_data(url, output_path):
    print(f"Attempting to download dataset from {url}...")
    try:
        # Request with a timeout to avoid hanging
        urllib.request.urlretrieve(url, output_path)
        print("Download successful!")
        return True
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        return False

def generate_synthetic_data(output_path, num_rows=5000):
    print("Generating synthetic TLC taxi data as fallback...")
    np.random.seed(42)
    
    # Generate random pickup times in 2017
    start_date = np.datetime64('2017-01-01T00:00:00')
    end_date = np.datetime64('2017-12-31T23:59:59')
    delta = end_date - start_date
    random_offsets = np.random.randint(0, delta.astype('int64'), size=num_rows)
    pickup_times = start_date + random_offsets.astype('timedelta64[s]')
    
    # Generate durations (normal distribution around 15 minutes, min 1 minute)
    durations_min = np.random.normal(loc=15, scale=8, size=num_rows)
    durations_min = np.clip(durations_min, a_min=1, a_max=120)
    durations_td = durations_min.astype('timedelta64[m]')
    dropoff_times = pickup_times + durations_td
    
    # Trip distances (exponential distribution, average 3 miles)
    trip_distances = np.random.exponential(scale=3.0, size=num_rows)
    trip_distances = np.clip(trip_distances, a_min=0.1, a_max=50.0)
    
    # PULocationID and DOLocationID
    # We group routes, so let's limit location IDs to a small set (e.g. 10 locations) to ensure overlap for grouped stats
    pu_ids = np.random.randint(1, 15, size=num_rows)
    do_ids = np.random.randint(1, 15, size=num_rows)
    
    # Payment type: 1 (Credit Card) ~70%, 2 (Cash) ~30%
    payment_types = np.random.choice([1, 2], size=num_rows, p=[0.7, 0.3])
    
    # Fare amounts
    # Base fare 2.50 + $2.50 per mile + normal noise
    fare_amounts = 2.50 + 2.50 * trip_distances + np.random.normal(loc=0, scale=2.0, size=num_rows)
    fare_amounts = np.clip(fare_amounts, a_min=2.50, a_max=150.0)
    
    # Extra, mta_tax, tolls, improvement surcharge
    extras = np.random.choice([0.0, 0.5, 1.0], size=num_rows, p=[0.6, 0.2, 0.2])
    mta_taxes = np.full(num_rows, 0.5)
    tolls_amounts = np.random.choice([0.0, 5.76], size=num_rows, p=[0.9, 0.1])
    improvement_surcharges = np.full(num_rows, 0.3)
    
    # Tip amounts
    # Cash payments have $0 tip. Credit cards have ~18% tip on average (some tip 20%+, some tip 0%)
    tip_amounts = np.zeros(num_rows)
    for i in range(num_rows):
        if payment_types[i] == 1:
            tip_rate = np.random.choice([0.0, 0.10, 0.15, 0.18, 0.20, 0.22, 0.25], p=[0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1])
            tip_amounts[i] = round(tip_rate * fare_amounts[i], 2)
            
    total_amounts = fare_amounts + extras + mta_taxes + tip_amounts + tolls_amounts + improvement_surcharges
    
    df = pd.DataFrame({
        "Unnamed: 0": np.arange(num_rows),
        "VendorID": np.random.choice([1, 2], size=num_rows),
        "tpep_pickup_datetime": pd.Series(pickup_times).dt.strftime('%m/%d/%Y %I:%M:%S %p'),
        "tpep_dropoff_datetime": pd.Series(dropoff_times).dt.strftime('%m/%d/%Y %I:%M:%S %p'),
        "passenger_count": np.random.randint(1, 6, size=num_rows),
        "trip_distance": np.round(trip_distances, 2),
        "RatecodeID": np.random.choice([1, 2, 5], size=num_rows, p=[0.95, 0.03, 0.02]),
        "store_and_fwd_flag": np.random.choice(["N", "Y"], size=num_rows, p=[0.99, 0.01]),
        "PULocationID": pu_ids,
        "DOLocationID": do_ids,
        "payment_type": payment_types,
        "fare_amount": np.round(fare_amounts, 2),
        "extra": extras,
        "mta_tax": mta_taxes,
        "tip_amount": tip_amounts,
        "tolls_amount": tolls_amounts,
        "improvement_surcharge": improvement_surcharges,
        "total_amount": np.round(total_amounts, 2)
    })
    
    df.to_csv(output_path, index=False)
    print(f"Synthetic data saved to {output_path}")

def ingest_data(config_path=None):
    config = load_config(config_path)
    
    raw_dir = config["paths"]["raw_data_dir"]
    os.makedirs(raw_dir, exist_ok=True)
    
    filename = config["paths"]["raw_data_filename"]
    output_path = os.path.join(raw_dir, filename)
    
    # GitHub raw download link for NYC taxi data subset (Automatidata)
    url = "https://raw.githubusercontent.com/sumitdgr/Automatidata_Project/master/2017_Yellow_Taxi_Trip_Data.csv"
    
    success = download_data(url, output_path)
    if not success:
        generate_synthetic_data(output_path)
        
    print("Data ingestion completed successfully.")

if __name__ == "__main__":
    ingest_data()
