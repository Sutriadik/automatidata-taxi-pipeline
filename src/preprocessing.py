import pandas as pd
import numpy as np

def parse_datetimes(df):
    """
    Parses datetime string columns in the TLC dataset into datetime objects.
    """
    df = df.copy()
    datetime_formats = ['%m/%d/%Y %I:%M:%S %p', '%Y-%m-%d %H:%M:%S']
    
    for col in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']:
        for dt_format in datetime_formats:
            try:
                df[col] = pd.to_datetime(df[col], format=dt_format)
                break
            except ValueError:
                continue
        # Fallback to general parsing if none matches
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col])
            
    return df

def outlier_imputer(df, columns, factor=6.0):
    """
    Caps values in the specified columns based on their interquartile range (IQR).
    Capping threshold: Q3 + (factor * IQR). Lower bound is clipped to 0.
    """
    df = df.copy()
    for col in columns:
        # Reassign values < 0 to 0
        df.loc[df[col] < 0, col] = 0.0
        
        # Calculate IQR and upper threshold
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_threshold = q3 + (factor * iqr)
        
        # Cap upper outliers
        df.loc[df[col] > upper_threshold, col] = upper_threshold
        print(f"Capping outliers for column '{col}': Q3={q3:.2f}, Threshold={upper_threshold:.2f}")
        
    return df

def create_target_labels(df):
    """
    Filters the dataframe to credit card transactions only and engineers
    the target tip classification variable: 'generous' (1 for tip >= 20%, 0 otherwise).
    """
    df = df.copy()
    
    # Isolate credit card users (payment_type == 1)
    df_cc = df[df['payment_type'] == 1].copy()
    
    # tip percent = tip_amount / (total_amount - tip_amount)
    # Avoid division by zero
    denominator = df_cc['total_amount'] - df_cc['tip_amount']
    df_cc['tip_percent'] = np.where(denominator > 0, df_cc['tip_amount'] / denominator, 0.0)
    df_cc['tip_percent'] = np.round(df_cc['tip_percent'], 3)
    
    # Target label: generous = tip_percent >= 20%
    df_cc['generous'] = (df_cc['tip_percent'] >= 0.20).astype(int)
    
    return df_cc
