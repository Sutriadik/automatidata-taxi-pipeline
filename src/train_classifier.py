import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

def train_tip_classifier(df_train, df_test, config):
    """
    Fits and saves the classification model for predicting generous tippers.
    Compares Random Forest and XGBoost using RandomizedSearchCV and selects the best by F1 score.
    """
    print("Starting Tip Classifier training...")
    
    # 1. Select features and target
    # Numeric features
    numeric_features = ['passenger_count', 'mean_distance', 'mean_duration', 'predicted_fare',
                        'am_rush', 'daytime', 'pm_rush', 'nighttime']
    
    # Categorical features
    categorical_features = ['VendorID', 'RatecodeID', 'PULocationID', 'DOLocationID', 'day', 'month']
    
    target = 'generous'
    
    # Separate features and target
    X_train = df_train[numeric_features + categorical_features].copy()
    y_train = df_train[target].values.ravel()
    
    X_test = df_test[numeric_features + categorical_features].copy()
    y_test = df_test[target].values.ravel()
    
    # Convert categorical features to string to ensure correct one-hot encoding
    for col in categorical_features:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)
        
    # 2. Define preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # 3. Model Search using RandomizedSearchCV
    # Random Forest Setup
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=config["preprocessing"]["random_state"]))
    ])
    
    # XGBoost Setup
    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(random_state=config["preprocessing"]["random_state"], eval_metric='logloss'))
    ])
    
    # Load parameters from config
    rf_grid = {"classifier__" + k: v for k, v in config["models"]["classifier"]["rf_params"].items()}
    xgb_grid = {"classifier__" + k: v for k, v in config["models"]["classifier"]["xgb_params"].items()}
    
    n_iter = config["tuning"]["n_iter"]
    cv = config["tuning"]["cv"]
    random_state = config["preprocessing"]["random_state"]
    
    print("Running Random Forest Randomized Search...")
    rf_search = RandomizedSearchCV(rf_pipeline, rf_grid, n_iter=n_iter, cv=cv, scoring='f1', n_jobs=-1, random_state=random_state)
    rf_search.fit(X_train, y_train)
    rf_best_score = rf_search.best_score_
    print(f"Random Forest Best CV F1 Score: {rf_best_score:.4f}")
    
    print("Running XGBoost Randomized Search...")
    xgb_search = RandomizedSearchCV(xgb_pipeline, xgb_grid, n_iter=n_iter, cv=cv, scoring='f1', n_jobs=-1, random_state=random_state)
    xgb_search.fit(X_train, y_train)
    xgb_best_score = xgb_search.best_score_
    print(f"XGBoost Best CV F1 Score: {xgb_best_score:.4f}")
    
    # 4. Select the best model
    if rf_best_score >= xgb_best_score:
        print("Selecting Random Forest as the final classifier model.")
        best_pipeline = rf_search.best_estimator_
    else:
        print("Selecting XGBoost as the final classifier model.")
        best_pipeline = xgb_search.best_estimator_
        
    # Evaluate best model on test set
    y_test_pred = best_pipeline.predict(X_test)
    test_f1 = f1_score(y_test, y_test_pred)
    print(f"Final Classifier Test F1 Score: {test_f1:.4f}")
    
    # 5. Save Model
    models_dir = config["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, config["paths"]["classifier_filename"])
    
    with open(model_path, "wb") as f:
        pickle.dump(best_pipeline, f)
    print(f"Tip Classifier saved to {model_path}")
    
    return best_pipeline
