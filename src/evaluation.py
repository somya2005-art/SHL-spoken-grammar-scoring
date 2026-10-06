import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_metrics(y_true, y_pred, prefix=""):
    """
    Computes all standard speech & grammar scoring evaluation metrics:
    - Pearson correlation coefficient (r)
    - RMSE (Root Mean Squared Error)
    - Spearman rank correlation (rho)
    - MAE (Mean Absolute Error)
    - R2 Score
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    # Clip any NaN or infinite values
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) < 2:
        return {}

    p_corr, _ = pearsonr(y_true, y_pred)
    s_corr, _ = spearmanr(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    pfx = f"{prefix}_" if prefix else ""
    return {
        f"{pfx}pearson": float(p_corr),
        f"{pfx}rmse": float(rmse),
        f"{pfx}spearman": float(s_corr),
        f"{pfx}mae": float(mae),
        f"{pfx}r2": float(r2)
    }

def print_metrics(metrics_dict, title="Validation Metrics"):
    """
    Pretty prints metrics.
    """
    print(f"\n==================== {title} ====================")
    for k, v in metrics_dict.items():
        print(f"  {k:25s}: {v:.4f}")
    print("====================================================\n")

def calibrate_predictions(y_train_true, y_train_pred, y_test_pred, clip_range=(0.0, 5.0)):
    """
    Fits an optimal linear regression calibration on OOF predictions and applies to test predictions.
    Also clips predictions to the valid Likert range [0.0, 5.0].
    """
    from sklearn.linear_model import Ridge
    calibrator = Ridge(alpha=1.0)
    calibrator.fit(y_train_pred.reshape(-1, 1), y_train_true)
    
    cal_oof = calibrator.predict(y_train_pred.reshape(-1, 1))
    cal_test = calibrator.predict(y_test_pred.reshape(-1, 1))

    if clip_range is not None:
        cal_oof = np.clip(cal_oof, clip_range[0], clip_range[1])
        cal_test = np.clip(cal_test, clip_range[0], clip_range[1])

    return cal_oof, cal_test, calibrator
