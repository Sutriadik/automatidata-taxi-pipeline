import os
import argparse
import pickle
import json
import pandas as pd
import numpy as np

def load_route_stats(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Predict Taxi Fare & Generous Tip Likelihood")
    parser.add_argument("--vendor", type=int, default=2, help="VendorID (1 or 2)")
    parser.add_argument("--passengers", type=int, default=1, help="Passenger Count")
    parser.add_argument("--distance", type=float, required=True, help="Trip Distance (miles)")
    parser.add_argument("--duration", type=float, required=True, help="Trip Duration (minutes)")
    parser.add_argument("--pu_id", type=int, default=100, help="PULocationID")
    parser.add_argument("--do_id", type=int, default=231, help="DOLocationID")
    parser.add_argument("--hour", type=int, required=True, help="Pickup Hour (0-23)")
    parser.add_argument("--day", type=str, required=True, help="Day of Week (e.g. monday, saturday)")
    parser.add_argument("--month", type=str, required=True, help="Month (e.g. march, december)")
    parser.add_argument("--rate_code", type=int, default=1, help="RatecodeID")
    
    args = parser.parse_args()
    
    # 1. Resolve paths relative to this script
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    regressor_path = os.path.join(project_root, "models", "fare_regressor.pkl")
    classifier_path = os.path.join(project_root, "models", "tip_classifier.pkl")
    route_stats_path = os.path.join(project_root, "data", "processed", "route_stats.json")
    
    # 2. Check if models exist
    if not (os.path.exists(regressor_path) and os.path.exists(classifier_path)):
        print("Error: Serialized models not found in models/ directory. Please run src/pipeline.py first to train models.")
        return
        
    # 3. Load Models and Route Stats
    with open(regressor_path, "rb") as f:
        reg_pipeline = pickle.load(f)
        
    with open(classifier_path, "rb") as f:
        clf_pipeline = pickle.load(f)
        
    route_stats = load_route_stats(route_stats_path)
    
    # 4. Map Route averages
    route_key = f"{args.pu_id} {args.do_id}"
    mean_distance = route_stats["grouped_dist"].get(route_key, route_stats["global_avg_dist"])
    mean_duration = route_stats["grouped_dur"].get(route_key, route_stats["global_avg_dur"])
    
    # 5. Extract time features
    day_lower = args.day.lower()
    month_lower = args.month.lower()[:3]  # abbreviate to 3 chars
    
    is_weekday = day_lower not in ["saturday", "sunday"]
    is_rush_time = (6 <= args.hour < 10) or (16 <= args.hour < 20)
    rush_hour = 1 if (is_weekday and is_rush_time) else 0
    
    am_rush = 1 if (6 <= args.hour < 10) else 0
    daytime = 1 if (10 <= args.hour < 16) else 0
    pm_rush = 1 if (16 <= args.hour < 20) else 0
    nighttime = 1 if (args.hour >= 20 or args.hour < 6) else 0
    
    # 6. Predict predicted_fare (Stage 1)
    df_reg = pd.DataFrame([{
        "VendorID": str(args.vendor),
        "passenger_count": args.passengers,
        "mean_distance": mean_distance,
        "mean_duration": mean_duration,
        "rush_hour": rush_hour
    }])
    
    predicted_fare = reg_pipeline.predict(df_reg)[0]
    
    # 7. Predict tip classification (Stage 2)
    df_clf = pd.DataFrame([{
        "passenger_count": args.passengers,
        "mean_distance": mean_distance,
        "mean_duration": mean_duration,
        "predicted_fare": predicted_fare,
        "am_rush": am_rush,
        "daytime": daytime,
        "pm_rush": pm_rush,
        "nighttime": nighttime,
        "VendorID": str(args.vendor),
        "RatecodeID": str(args.rate_code),
        "PULocationID": str(args.pu_id),
        "DOLocationID": str(args.do_id),
        "day": day_lower,
        "month": month_lower
    }])
    
    is_generous = clf_pipeline.predict(df_clf)[0]
    prob_generous = clf_pipeline.predict_proba(df_clf)[0][1]
    
    # 8. Display output beautifully
    print("="*50)
    print("      AUTOMATIDATA TRIP INFERENCE PREDICTIONS")
    print("="*50)
    print(f"Inputs:")
    print(f"  - Route             : PULocation {args.pu_id} -> DOLocation {args.do_id}")
    print(f"  - Day & Time        : {args.day.capitalize()} at {args.hour:02d}:00")
    print(f"  - Distance / Duration: {args.distance:.2f} miles / {args.duration:.2f} minutes")
    print("-"*50)
    print(f"Predictions:")
    print(f"  * Estimated Trip Fare      : ${predicted_fare:.2f}")
    
    generous_label = "YES (Likely >= 20%)" if is_generous == 1 else "NO (Likely < 20%)"
    print(f"  * Generous Tipper Predicted: {generous_label}")
    print(f"  * Generous Tip Probability : {prob_generous*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
