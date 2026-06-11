import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from preprocessing import parse_datetimes, outlier_imputer
from feature_engineering import compute_route_stats, apply_route_stats, create_time_features, save_route_stats

def train_fare_regressor(df, config):
    """
    Fits and saves the best non-linear regressor model for predicting fare amounts.
    Compares Random Forest Regressor and XGBoost Regressor using RandomizedSearchCV.
    """
    print("Starting non-linear Fare Regressor training...")
    
    # 1. Parse datetimes and compute duration on the full dataset
    df = parse_datetimes(df)
    df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']) / np.timedelta64(1, 'm')
    df.loc[df['duration'] < 0, 'duration'] = 0.0
    
    # 2. Outlier capping on the entire dataset to ensure clean evaluation on test set
    iqr_factor = config["preprocessing"]["iqr_factor"]
    df = outlier_imputer(df, ['fare_amount', 'duration'], iqr_factor)
    
    # Train-test split
    val_split = config["preprocessing"]["validation_split"]
    random_state = config["preprocessing"]["random_state"]
    df_train, df_test = train_test_split(df, test_size=val_split, random_state=random_state)
    
    # 3. Compute route stats from training set only
    stats = compute_route_stats(df_train)
    processed_dir = config["paths"]["processed_data_dir"]
    os.makedirs(processed_dir, exist_ok=True)
    save_route_stats(stats, os.path.join(processed_dir, config["paths"]["route_stats_filename"]))
    
    # 4. Map route stats to both datasets
    df_train = apply_route_stats(df_train, stats)
    df_test = apply_route_stats(df_test, stats)
    
    # 5. Day and rush hour extraction
    df_train = create_time_features(df_train)
    df_test = create_time_features(df_test)
    
    # 6. Separate features and target
    features = ['VendorID', 'passenger_count', 'mean_distance', 'mean_duration', 'rush_hour']
    target = 'fare_amount'
    
    X_train = df_train[features].copy()
    y_train = df_train[target].values.ravel()
    
    X_test = df_test[features].copy()
    y_test = df_test[target].values.ravel()
    
    # Ensure VendorID is treated as categorical
    X_train['VendorID'] = X_train['VendorID'].astype(str)
    X_test['VendorID'] = X_test['VendorID'].astype(str)
    
    # 7. Create preprocessing pipeline
    numeric_features = ['passenger_count', 'mean_distance', 'mean_duration', 'rush_hour']
    categorical_features = ['VendorID']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ]
    )
    
    # 8. Model Search using RandomizedSearchCV
    # Random Forest Setup
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=random_state))
    ])
    
    # XGBoost Setup
    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(random_state=random_state))
    ])
    
    # Load parameters from config
    rf_grid = {"regressor__" + k: v for k, v in config["models"]["regressor"]["rf_params"].items()}
    xgb_grid = {"regressor__" + k: v for k, v in config["models"]["regressor"]["xgb_params"].items()}
    
    n_iter = config["tuning"]["n_iter"]
    cv = config["tuning"]["cv"]
    
    print("Running Random Forest Regressor Search...")
    rf_search = RandomizedSearchCV(rf_pipeline, rf_grid, n_iter=n_iter, cv=cv, scoring='r2', n_jobs=-1, random_state=random_state)
    rf_search.fit(X_train, y_train)
    rf_best_score = rf_search.best_score_
    print(f"Random Forest Regressor Best CV R2: {rf_best_score:.4f}")
    
    print("Running XGBoost Regressor Search...")
    xgb_search = RandomizedSearchCV(xgb_pipeline, xgb_grid, n_iter=n_iter, cv=cv, scoring='r2', n_jobs=-1, random_state=random_state)
    xgb_search.fit(X_train, y_train)
    xgb_best_score = xgb_search.best_score_
    print(f"XGBoost Regressor Best CV R2: {xgb_best_score:.4f}")
    
    # Select the best model
    if rf_best_score >= xgb_best_score:
        print("Selecting Random Forest Regressor as the final regressor.")
        best_pipeline = rf_search.best_estimator_
        best_score = rf_best_score
    else:
        print("Selecting XGBoost Regressor as the final regressor.")
        best_pipeline = xgb_search.best_estimator_
        best_score = xgb_best_score
        
    y_test_pred = best_pipeline.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    print(f"Final Regressor Test R2 Score: {test_r2:.4f}")
    
    # 9. Save model
    models_dir = config["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, config["paths"]["regressor_filename"])
    
    with open(model_path, "wb") as f:
        pickle.dump(best_pipeline, f)
    print(f"Fare Regressor saved to {model_path}")
    
    return best_pipeline, df_train, df_test
