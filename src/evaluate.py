import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
)

def evaluate_regressor(y_true, y_pred, output_dir="models"):
    """
    Computes regression metrics and saves the actual vs predicted plot.
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    print("\n" + "="*40)
    print("REGRESSION MODEL PERFORMANCE METRICS")
    print("="*40)
    print(f"R^2 Coefficient of Determination: {r2:.4f}")
    print(f"Mean Absolute Error (MAE)       : ${mae:.2f}")
    print(f"Mean Squared Error (MSE)        : {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE)  : ${rmse:.2f}")
    print("="*40)
    
    # Plot Actual vs Predicted
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(7, 7))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.3)
    # 45 degree diagonal line
    max_val = max(max(y_true), max(y_pred))
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--')
    plt.xlabel("Actual Fare ($)")
    plt.ylabel("Predicted Fare ($)")
    plt.title("Actual vs. Predicted Taxi Fares")
    plot_path = os.path.join(output_dir, "regression_actual_vs_predicted.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved regression performance plot to {plot_path}")
    
    return {"r2": r2, "mae": mae, "rmse": rmse}

def evaluate_classifier(y_true, y_pred, y_prob, output_dir="models"):
    """
    Computes classification metrics and saves confusion matrix and ROC curve plots.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print("\n" + "="*40)
    print("CLASSIFICATION MODEL PERFORMANCE METRICS")
    print("="*40)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("="*40)
    
    # 1. Confusion Matrix Plot
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Generous', 'Generous'], yticklabels=['Not Generous', 'Generous'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Generous Tipper Confusion Matrix')
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")
    
    # 2. ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve plot to {roc_path}")
    
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": roc_auc}
