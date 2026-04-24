import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def forecast_error_overtime_plot(y_true, y_pred):
    plt.figure(figsize=(12, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Perfect Prediction')
    plt.title("Predicted vs Actual RUL")
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.legend()
    plt.grid()
    plt.show()

def residuals_plot(residuals, y_pred):
    plt.figure(figsize=(12, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residuals vs Predicted Values")
    plt.xlabel("Predicted RUL")
    plt.ylabel("Residuals")
    plt.grid()
    plt.show()

def residuals_histogram(residuals):
    plt.figure(figsize=(12, 6))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title("Distribution of Residuals")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()

##########################################################################
# for use with final models #
##########################################################################
def residuals_analysis(y_true, y_pred):
    # for use on final models after tuning
    residuals = y_true - y_pred
    residuals_plot(residuals, y_pred)
    residuals_histogram(residuals)

def evaluate_model(model, X_test, y_test):
    # for use on final models after tuning
    predictions = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, predictions)
    mse = mean_absolute_error(y_test, predictions)
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mse}") 
    display(model.feature_importances_)
    forecast_error_overtime_plot(y_test, predictions)


##########################################################################
# for use in hyperparameter tuning #
##########################################################################
def evaluate_model_numerics(model, X_test, y_test):
    """Calculates standard regression metrics and returns them as a dictionary."""
    predictions = model.predict(X_test)
    
    # Calculate metrics
    rmse = root_mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2_score": float(r2)
    }
    
    return metrics, predictions

def residuals_analysis_numerics(y_true, y_pred):
    """Quantifies residual behavior using statistical measures."""
    # Ensure inputs are flat arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    residuals = y_true - y_pred
    
    res_mean = np.mean(residuals)
    res_std = np.std(residuals)
    
    # Skewness: > 0 means underestimating RUL, < 0 means overestimating 
    res_skew = pd.Series(residuals).skew()
    
    # Calculate % of predictions within a 10-cycle margin
    within_margin = np.mean(np.abs(residuals) <= 10) * 100
    
    res_metrics = {
        "res_mean": float(res_mean),
        "res_std": float(res_std),
        "res_skew": float(res_skew),
        "pct_within_10_cycles": float(within_margin)
    }
    
    return res_metrics
