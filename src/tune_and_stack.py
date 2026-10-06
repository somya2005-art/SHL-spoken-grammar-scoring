import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, HuberRegressor, ElasticNet
from sklearn.feature_selection import SelectPercentile, f_regression
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from src.evaluation import compute_metrics, print_metrics, calibrate_predictions
import warnings
warnings.filterwarnings("ignore")

def run_advanced_stacking_pipeline():
    feat_dir = r"C:\Users\Somya\.gemini\antigravity\scratch\shl_grammar_scoring\data\features"
    train_df = pd.read_csv(os.path.join(feat_dir, "engineered_features_train.csv"))
    test_df = pd.read_csv(os.path.join(feat_dir, "engineered_features_test.csv"))

    feature_cols = [c for c in train_df.columns if c not in ['filename', 'label']]
    print(f"Initial features: {len(feature_cols)}")

    X = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = np.clip(X, -1e6, 1e6)
    y = train_df['label'].values
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test = np.clip(X_test, -1e6, 1e6)

    # 1. Feature Selection: Remove zero-variance or near-zero variance features
    std_devs = X.std(axis=0)
    valid_cols = std_devs[std_devs > 1e-6].index.tolist()
    X = X[valid_cols]
    X_test = X_test[valid_cols]
    print(f"Features after zero-variance filtering: {len(valid_cols)}")

    # 5-Fold Stratified Split
    y_binned = np.round(y * 2).astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(X, y_binned))

    # -------------------------------------------------------------
    # Model 1: Tuned LightGBM (Huber / L2 Objective)
    # -------------------------------------------------------------
    print("\n--- Training Tuned LightGBM ---")
    lgb_oof = np.zeros(len(X))
    lgb_test = np.zeros(len(X_test))

    lgb_params = {
        'objective': 'huber',
        'huber_alpha': 0.9,
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': 750,
        'learning_rate': 0.025,
        'num_leaves': 14,
        'max_depth': 4,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'reg_alpha': 0.3,
        'reg_lambda': 2.0,
        'min_child_samples': 18,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }

    for fold, (trn_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X.iloc[trn_idx], y[trn_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        lgb_oof[val_idx] = model.predict(X_va)
        lgb_test += model.predict(X_test) / len(folds)

    m_lgb = compute_metrics(y, lgb_oof, prefix="Tuned_LGBM")
    print_metrics(m_lgb, "Tuned LightGBM Metrics")

    # -------------------------------------------------------------
    # Model 2: Tuned XGBoost
    # -------------------------------------------------------------
    print("\n--- Training Tuned XGBoost ---")
    xgb_oof = np.zeros(len(X))
    xgb_test = np.zeros(len(X_test))

    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': 600,
        'learning_rate': 0.025,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.4,
        'reg_lambda': 2.0,
        'random_state': 42,
        'n_jobs': -1
    }

    for fold, (trn_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X.iloc[trn_idx], y[trn_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        xgb_oof[val_idx] = model.predict(X_va)
        xgb_test += model.predict(X_test) / len(folds)

    m_xgb = compute_metrics(y, xgb_oof, prefix="Tuned_XGB")
    print_metrics(m_xgb, "Tuned XGBoost Metrics")

    # -------------------------------------------------------------
    # Model 3: Tuned CatBoost
    # -------------------------------------------------------------
    print("\n--- Training Tuned CatBoost ---")
    cat_oof = np.zeros(len(X))
    cat_test = np.zeros(len(X_test))

    cat_params = {
        'iterations': 750,
        'learning_rate': 0.025,
        'depth': 4,
        'l2_leaf_reg': 5.0,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 42,
        'verbose': False
    }

    for fold, (trn_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X.iloc[trn_idx], y[trn_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        model = CatBoostRegressor(**cat_params)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False)
        cat_oof[val_idx] = model.predict(X_va)
        cat_test += model.predict(X_test) / len(folds)

    m_cat = compute_metrics(y, cat_oof, prefix="Tuned_CatBoost")
    print_metrics(m_cat, "Tuned CatBoost Metrics")

    # -------------------------------------------------------------
    # Model 4: Robust Huber Regressor (Linear robust)
    # -------------------------------------------------------------
    print("\n--- Training Huber Regressor ---")
    huber_oof = np.zeros(len(X))
    huber_test = np.zeros(len(X_test))

    for fold, (trn_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X.iloc[trn_idx], y[trn_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        scaler = RobustScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        X_te_sc = scaler.transform(X_test)

        model = HuberRegressor(alpha=10.0, max_iter=500)
        model.fit(X_tr_sc, y_tr)

        huber_oof[val_idx] = model.predict(X_va_sc)
        huber_test += model.predict(X_te_sc) / len(folds)

    m_hub = compute_metrics(y, huber_oof, prefix="Huber")
    print_metrics(m_hub, "Huber Regressor Metrics")

    # -------------------------------------------------------------
    # Stacking Meta-Learner & Optimal Blending
    # -------------------------------------------------------------
    print("\n--- Optimizing Final Stacking Ensemble ---")
    oof_candidates = [lgb_oof, xgb_oof, cat_oof, huber_oof]
    test_candidates = [lgb_test, xgb_test, cat_test, huber_test]
    names = ['Tuned_LGBM', 'Tuned_XGB', 'Tuned_CatBoost', 'Huber']

    oof_mat = np.column_stack(oof_candidates)

    def ensemble_objective(weights):
        weights = weights / np.sum(weights)
        pred = np.dot(oof_mat, weights)
        corr = compute_metrics(y, pred)['pearson']
        rmse = compute_metrics(y, pred)['rmse']
        return -corr + 0.35 * rmse

    init_w = np.ones(len(names)) / len(names)
    bnds = [(0.0, 1.0) for _ in range(len(names))]
    cons = ({'type': 'eq', 'fun': lambda w: 1.0 - np.sum(w)})

    opt_res = minimize(ensemble_objective, init_w, method='SLSQP', bounds=bnds, constraints=cons)
    weights = opt_res.x / np.sum(opt_res.x)

    print("Optimal Stacking Ensemble Weights:")
    for name, w in zip(names, weights):
        print(f"  {name:20s}: {w*100:.1f}%")

    ensemble_oof = np.dot(oof_mat, weights)
    ensemble_test = np.dot(np.column_stack(test_candidates), weights)

    # Calibration
    cal_oof, cal_test, calibrator = calibrate_predictions(y, ensemble_oof, ensemble_test, clip_range=(0.0, 5.0))
    final_metrics = compute_metrics(y, cal_oof, prefix="Stacking_Calibrated")
    print_metrics(final_metrics, "Final Calibrated Stacking Ensemble Metrics")

    # Save submission
    sub_dir = r"C:\Users\Somya\.gemini\antigravity\scratch\shl_grammar_scoring\submissions"
    sub_path = os.path.join(sub_dir, "submission_v3_tuned_stacking_ensemble.csv")
    sub_df = pd.DataFrame({
        'filename': test_df['filename'],
        'label': np.clip(cal_test, 0.0, 5.0)
    })
    sub_df.to_csv(sub_path, index=False)
    print(f"Generated Submission: {sub_path}")
    print(f"  Rows: {len(sub_df)}, Nulls: {sub_df.isnull().sum().to_dict()}, Range: [{sub_df['label'].min():.3f}, {sub_df['label'].max():.3f}]")

    return final_metrics, sub_path

if __name__ == "__main__":
    run_advanced_stacking_pipeline()
