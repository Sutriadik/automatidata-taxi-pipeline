# Automatidata NYC Taxi ML Pipeline

This repository contains a modular, dual-stage predictive pipeline for NYC Yellow Taxi trips using the **Automatidata** dataset. It predicts the trip fare amount and classifies whether a passenger is likely to leave a generous tip (>= 20%).

## 🎯 Project Overview

This project implements a complete, production-grade machine learning workflow split into two logical stages:
1. **Stage 1 (Regressor)**: Predicts the estimated fare amount based on route history, trip duration, distance, and temporal bins using an optimized **XGBoost Regressor**.
2. **Stage 2 (Classifier)**: Classifies whether credit card passengers will tip generously ($\ge$ 20%) using the predicted fare amount from Stage 1 along with pickup/dropoff location rates, times of day, and trip details via an optimized **Random Forest Classifier**.

---

## 📁 Project Structure

```text
automatidata_pipeline/
├── config/
│   └── config.yaml           # Model hyperparameters & search space settings
├── data/
│   ├── raw/                  # Downloaded raw dataset
│   └── processed/            # Preprocessed credit card training data & route statistics
├── models/                   # Serialized model pipelines and evaluation plots
├── src/
│   ├── data_ingestion.py     # Pulls raw TLC dataset or generates synthetic fallback
│   ├── preprocessing.py      # Datetime parsing, outlier capping, and target labeling
│   ├── feature_engineering.py# Computes route statistics, time-of-day bins, rush hour flags
│   ├── train_regressor.py    # RandomizedSearchCV tuning for Regressors
│   ├── train_classifier.py   # RandomizedSearchCV tuning for Classifiers
│   ├── evaluate.py           # Evaluation helper saving matrices and plots
│   └── pipeline.py           # Orchestration runner
├── requirements.txt          # Python dependency list
├── .gitignore                # Excludes data/models/caches from git tracking
└── predict.py                # Command-line interface for inference
```

---

## ⚙️ Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sutriadik/automatidata-taxi-pipeline.git
   cd automatidata-taxi-pipeline
   ```

2. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📈 Model Performance & Metrics

The pipeline prevents data leakage by calculating route statistics (mean distance/duration) strictly on the training fold. Below are the metrics computed on the unseen test set:

### Stage 1: Fare Regressor (XGBoost)
*   **$R^2$ Score**: `0.6682`
*   **Mean Absolute Error (MAE)**: `$3.07`
*   **Root Mean Squared Error (RMSE)**: `$5.86`

### Stage 2: Tip Classifier (Random Forest)
*   **Accuracy**: `69.57%`
*   **Precision**: `68.33%`
*   **Recall**: `78.61%`
*   **F1-Score**: `73.11%`

---

## 🚀 Usage

### 1. Run the Entire ML Pipeline
Run the master script to ingest the raw NYC TLC dataset, run outlier capping, compute route statistics, perform hyperparameter tuning, evaluate models, and save output metrics/plots:
```bash
python3 src/pipeline.py
```

### 2. Make Real-Time Predictions via CLI
Use the serialized model pipelines to make predictions for new trips. Provide pickup/dropoff parameters:
```bash
python3 predict.py --vendor 2 --passengers 1 --distance 3.5 --duration 20 --hour 18 --day monday --month march
```

**Output example:**
```text
==================================================
      AUTOMATIDATA TRIP INFERENCE PREDICTIONS
==================================================
Inputs:
  - Route             : PULocation 100 -> DOLocation 231
  - Day & Time        : Monday at 18:00
  - Distance / Duration: 3.50 miles / 20.00 minutes
--------------------------------------------------
Predictions:
  * Estimated Trip Fare      : $17.23
  * Generous Tipper Predicted: YES (Likely >= 20%)
  * Generous Tip Probability : 68.62%
==================================================
```
