import os
import pandas as pd
import numpy as np

from data_ingestion import load_config, ingest_data
from preprocessing import create_target_labels
from train_regressor import train_fare_regressor
from train_classifier import train_tip_classifier
from evaluate import evaluate_regressor, evaluate_classifier

def run_full_pipeline(config_path=None):
    print("="*60)
    print("STARTING AUTOMATIDATA TAXI ML PIPELINE")
    print("="*60)
    
    # 1. Load configuration
    config = load_config(config_path)
    
    # 2. Check and ingest data if missing
    raw_dir = config["paths"]["raw_data_dir"]
    filename = config["paths"]["raw_data_filename"]
    raw_data_path = os.path.join(raw_dir, filename)
    
    if not os.path.exists(raw_data_path):
        print(f"Raw data file not found at {raw_data_path}. Running data ingestion...")
        ingest_data(config_path)
    else:
        print(f"Found raw data file at {raw_data_path}.")
        
    # Load dataset
    df = pd.read_csv(raw_data_path)
    print(f"Loaded raw dataset of shape {df.shape}")
    
    # 3. Fit Regressor (Stage 1)
    reg_pipeline, df_train, df_test = train_fare_regressor(df, config)
    
    # 4. Predict predicted_fare on train/test sets to use as feature in Stage 2 classification
    features_reg = ['VendorID', 'passenger_count', 'mean_distance', 'mean_duration', 'rush_hour']
    
    # Prepare features ensuring correct VendorID category string type
    X_train_reg = df_train[features_reg].copy()
    X_train_reg['VendorID'] = X_train_reg['VendorID'].astype(str)
    
    X_test_reg = df_test[features_reg].copy()
    X_test_reg['VendorID'] = X_test_reg['VendorID'].astype(str)
    
    df_train['predicted_fare'] = reg_pipeline.predict(X_train_reg)
    df_test['predicted_fare'] = reg_pipeline.predict(X_test_reg)
    
    # Evaluate Regressor on test set
    y_test_reg = df_test['fare_amount'].values.ravel()
    evaluate_regressor(y_test_reg, df_test['predicted_fare'], config["paths"]["models_dir"])
    
    # 5. Isolate credit card payment subset and construct target labels
    print("Filtering credit card transactions and engineering targets...")
    df_train_cc = create_target_labels(df_train)
    df_test_cc = create_target_labels(df_test)
    print(f"Credit Card subsets - Train: {df_train_cc.shape[0]} rows, Test: {df_test_cc.shape[0]} rows")
    
    # 6. Fit Tip Classifier (Stage 2)
    clf_pipeline = train_tip_classifier(df_train_cc, df_test_cc, config)
    
    # 7. Evaluate Tip Classifier on test set
    # Define features for prediction
    numeric_features_clf = ['passenger_count', 'mean_distance', 'mean_duration', 'predicted_fare',
                            'am_rush', 'daytime', 'pm_rush', 'nighttime']
    categorical_features_clf = ['VendorID', 'RatecodeID', 'PULocationID', 'DOLocationID', 'day', 'month']
    
    X_test_clf = df_test_cc[numeric_features_clf + categorical_features_clf].copy()
    for col in categorical_features_clf:
        X_test_clf[col] = X_test_clf[col].astype(str)
        
    y_test_clf = df_test_cc['generous'].values.ravel()
    
    y_test_clf_pred = clf_pipeline.predict(X_test_clf)
    y_test_clf_prob = clf_pipeline.predict_proba(X_test_clf)[:, 1]
    
    evaluate_classifier(y_test_clf, y_test_clf_pred, y_test_clf_prob, config["paths"]["models_dir"])
    
    # 8. Save final processed training dataset for documentation/EDA
    processed_dir = config["paths"]["processed_data_dir"]
    os.makedirs(processed_dir, exist_ok=True)
    
    processed_path = os.path.join(processed_dir, config["paths"]["processed_data_filename"])
    df_train_cc.to_csv(processed_path, index=False)
    print(f"Saved processed credit-card training dataset to {processed_path}")
    
    print("\n" + "="*60)
    print("AUTOMATIDATA TAXI ML PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    # If run directly, locate config relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "config", "config.yaml")
    run_full_pipeline(config_path)
