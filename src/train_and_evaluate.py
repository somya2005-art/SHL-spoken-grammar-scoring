import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, ElasticNet
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from src.evaluation import compute_metrics, print_metrics, calibrate_predictions
import warnings
warnings.filterwarnings("ignore")

def run_experiment_pipeline(train_feat_path, test_feat_path, exp_name="v1_acoustic_baseline"):
    """
    End-to-end Cross-Validation, Model Training, Ensembling, and Submission Generation.
    """
    print(f"\n=======================================================")
    print(f"       RUNNING PIPELINE: {exp_name.upper()}           ")
    print(f"=======================================================")

    train_df = pd.read_csv(train_feat_path)
    test_df = pd.read_csv(test_feat_path)

    feature_cols = [c for c in train_df.columns if c not in ['filename', 'label']]
    print(f"Loaded {len(train_df)} train samples, {len(test_df)} test samples, {len(feature_cols)} features.")

    X = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Clip any extreme outliers
    X = np.clip(X, -1e6, 1e6)
    y = train_df['label'].values
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test = np.clip(X_test, -1e6, 1e6)

    # 5-Fold Stratified K-Fold based on binned continuous labels
    y_binned = np.round(y * 2).astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(X, y_binned))

    models_results = {}

    # -------------------------------------------------------------
    # Model 1: LightGBM Regressor
    # -------------------------------------------------------------
    print("\n--- Training LightGBM Regressor (5-Fold CV) ---")
    lgb_oof = np.zeros(len(X))
    lgb_test = np.zeros(len(X_test))
    lgb_feat_imp = np.zeros(len(feature_cols))

    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': 600,
        'learning_rate': 0.03,
        'num_leaves': 15,
        'max_depth': 4,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'reg_alpha': 0.2,
        'reg_lambda': 1.5,
        'min_child_samples': 15,
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
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
        )
        lgb_oof[val_idx] = model.predict(X_va)
        lgb_test += model.predict(X_test) / len(folds)
        lgb_feat_imp += model.feature_importances_ / len(folds)

    models_results['LightGBM'] = {
        'oof': lgb_oof,
        'test': lgb_test,
        'metrics': compute_metrics(y, lgb_oof, prefix="LGBM"),
        'feat_imp': lgb_feat_imp
    }
    print_metrics(models_results['LightGBM']['metrics'], "LightGBM 5-Fold OOF Metrics")

    # -------------------------------------------------------------
    # Model 2: XGBoost Regressor
    # -------------------------------------------------------------
    print("\n--- Training XGBoost Regressor (5-Fold CV) ---")
    xgb_oof = np.zeros(len(X))
    xgb_test = np.zeros(len(X_test))
    xgb_feat_imp = np.zeros(len(feature_cols))

    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': 500,
        'learning_rate': 0.03,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.2,
        'reg_lambda': 1.5,
        'random_state': 42,
        'n_jobs': -1
    }

    for fold, (trn_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X.iloc[trn_idx], y[trn_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        model = xgb.XGBRegressor(**xgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False
        )
        xgb_oof[val_idx] = model.predict(X_va)
        xgb_test += model.predict(X_test) / len(folds)
        xgb_feat_imp += model.feature_importances_ / len(folds)

    models_results['XGBoost'] = {
        'oof': xgb_oof,
        'test': xgb_test,
        'metrics': compute_metrics(y, xgb_oof, prefix="XGB"),
        'feat_imp': xgb_feat_imp
    }
    print_metrics(models_results['XGBoost']['metrics'], "XGBoost 5-Fold OOF Metrics")

    # -------------------------------------------------------------
    # Model 3: CatBoost Regressor
    # -------------------------------------------------------------
    print("\n--- Training CatBoost Regressor (5-Fold CV) ---")
    cat_oof = np.zeros(len(X))
    cat_test = np.zeros(len(X_test))
    cat_feat_imp = np.zeros(len(feature_cols))

    cat_params = {
        'iterations': 600,
        'learning_rate': 0.03,
        'depth': 4,
        'l2_leaf_reg': 3.0,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 42,
        'verbose': False
    }

    for fold, (trn_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X.iloc[trn_idx], y[trn_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        model = CatBoostRegressor(**cat_params)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=40,
            verbose=False
        )
        cat_oof[val_idx] = model.predict(X_va)
        cat_test += model.predict(X_test) / len(folds)
        cat_feat_imp += model.get_feature_importance() / len(folds)

    models_results['CatBoost'] = {
        'oof': cat_oof,
        'test': cat_test,
        'metrics': compute_metrics(y, cat_oof, prefix="CatBoost"),
        'feat_imp': cat_feat_imp
    }
    print_metrics(models_results['CatBoost']['metrics'], "CatBoost 5-Fold OOF Metrics")

    # -------------------------------------------------------------
    # Model 4: Regularized Ridge Regressor (Linear baseline)
    # -------------------------------------------------------------
    print("\n--- Training Regularized Ridge Regressor (5-Fold CV) ---")
    ridge_oof = np.zeros(len(X))
    ridge_test = np.zeros(len(X_test))

    for fold, (trn_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X.iloc[trn_idx], y[trn_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        X_te_sc = scaler.transform(X_test)

        model = Ridge(alpha=25.0, random_state=42)
        model.fit(X_tr_sc, y_tr)

        ridge_oof[val_idx] = model.predict(X_va_sc)
        ridge_test += model.predict(X_te_sc) / len(folds)

    models_results['Ridge'] = {
        'oof': ridge_oof,
        'test': ridge_test,
        'metrics': compute_metrics(y, ridge_oof, prefix="Ridge")
    }
    print_metrics(models_results['Ridge']['metrics'], "Ridge 5-Fold OOF Metrics")

    # -------------------------------------------------------------
    # Model 5: Weighted Multi-Model Ensemble
    # -------------------------------------------------------------
    print("\n--- Optimizing Diverse Multi-Model Ensemble ---")
    oof_candidates = [lgb_oof, xgb_oof, cat_oof, ridge_oof]
    test_candidates = [lgb_test, xgb_test, cat_test, ridge_test]
    names = ['LightGBM', 'XGBoost', 'CatBoost', 'Ridge']

    oof_mat = np.column_stack(oof_candidates)

    def ensemble_objective(weights):
        weights = weights / np.sum(weights)
        pred = np.dot(oof_mat, weights)
        corr, _ = compute_metrics(y, pred)['pearson'], 0
        rmse = compute_metrics(y, pred)['rmse']
        return -corr + 0.35 * rmse

    init_w = np.ones(len(names)) / len(names)
    bnds = [(0.0, 1.0) for _ in range(len(names))]
    cons = ({'type': 'eq', 'fun': lambda w: 1.0 - np.sum(w)})

    opt_res = minimize(ensemble_objective, init_w, method='SLSQP', bounds=bnds, constraints=cons)
    weights = opt_res.x / np.sum(opt_res.x)

    print("Optimal Ensemble Weights:")
    for name, w in zip(names, weights):
        print(f"  {name:15s}: {w*100:.1f}%")

    ensemble_oof = np.dot(oof_mat, weights)
    ensemble_test = np.dot(np.column_stack(test_candidates), weights)

    # Post-processing / Calibration
    cal_oof, cal_test, calibrator = calibrate_predictions(y, ensemble_oof, ensemble_test, clip_range=(0.0, 5.0))
    ens_metrics = compute_metrics(y, cal_oof, prefix="Ensemble_Calibrated")
    print_metrics(ens_metrics, "Final Calibrated Ensemble 5-Fold OOF Metrics")

    # -------------------------------------------------------------
    # Generate Submissions
    # -------------------------------------------------------------
    sub_dir = r"C:\Users\Somya\.gemini\antigravity\scratch\shl_grammar_scoring\submissions"
    os.makedirs(sub_dir, exist_ok=True)

    # 1. Best Single Model Submission (LightGBM)
    sub_single = pd.DataFrame({
        'filename': test_df['filename'],
        'label': np.clip(lgb_test, 0.0, 5.0)
    })
    sub_single_path = os.path.join(sub_dir, f"submission_{exp_name}_single_lgbm.csv")
    sub_single.to_csv(sub_single_path, index=False)
    print(f"Generated Single Model Submission: {sub_single_path}")

    # 2. Calibrated Ensemble Submission
    sub_ensemble = pd.DataFrame({
        'filename': test_df['filename'],
        'label': np.clip(cal_test, 0.0, 5.0)
    })
    sub_ensemble_path = os.path.join(sub_dir, f"submission_{exp_name}_ensemble.csv")
    sub_ensemble.to_csv(sub_ensemble_path, index=False)
    print(f"Generated Calibrated Ensemble Submission: {sub_ensemble_path}")

    # Verify submission formatting
    print("\nSubmission File Verification:")
    print(f"  Row count: {len(sub_ensemble)} (expected 216)")
    print(f"  Null count: {sub_ensemble.isnull().sum().to_dict()}")
    print(f"  Predicted Label range: [{sub_ensemble['label'].min():.3f}, {sub_ensemble['label'].max():.3f}]")
    print("  First 5 rows:")
    print(sub_ensemble.head(5))

    # -------------------------------------------------------------
    # Visualizations & Diagnostics for Final Report / Notebook
    # -------------------------------------------------------------
    fig_dir = r"C:\Users\Somya\.gemini\antigravity\scratch\shl_grammar_scoring\figures"
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Actual vs Predicted Scatter Plot
    plt.figure(figsize=(7, 6))
    plt.scatter(y, cal_oof, alpha=0.5, color='#1f77b4', edgecolors='k', s=40)
    plt.plot([0, 5], [0, 5], 'r--', lw=2, label='Perfect Prediction (y=x)')
    plt.xlabel('Ground Truth MOS Grammar Score', fontsize=12)
    plt.ylabel('OOF Predicted Grammar Score', fontsize=12)
    plt.title(f'Ground Truth vs OOF Predictions ({exp_name})\nPearson r={ens_metrics["Ensemble_Calibrated_pearson"]:.4f}, RMSE={ens_metrics["Ensemble_Calibrated_rmse"]:.4f}', fontsize=12)
    plt.legend(frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{exp_name}_actual_vs_pred.png"), dpi=300)
    plt.close()

    # 2. Top 15 Feature Importances
    top_indices = np.argsort(lgb_feat_imp)[-15:]
    top_features = [feature_cols[i] for i in top_indices]
    top_scores = lgb_feat_imp[top_indices]

    plt.figure(figsize=(9, 6))
    plt.barh(range(len(top_features)), top_scores, color='#2ca02c')
    plt.yticks(range(len(top_features)), top_features, fontsize=10)
    plt.xlabel('Average Feature Importance (LightGBM)', fontsize=12)
    plt.title(f'Top 15 Most Discriminative Features ({exp_name})', fontsize=13)
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{exp_name}_top_features.png"), dpi=300)
    plt.close()

    # 3. Model Benchmark Comparison Table
    summary_data = []
    for name, res in models_results.items():
        summary_data.append({
            'Model': name,
            'Pearson (r)': res['metrics'][f"{name}_pearson"] if f"{name}_pearson" in res['metrics'] else res['metrics'][list(res['metrics'].keys())[0]],
            'RMSE': res['metrics'][f"{name}_rmse"] if f"{name}_rmse" in res['metrics'] else res['metrics'][list(res['metrics'].keys())[1]],
            'Spearman (rho)': res['metrics'][f"{name}_spearman"] if f"{name}_spearman" in res['metrics'] else res['metrics'][list(res['metrics'].keys())[2]],
            'MAE': res['metrics'][f"{name}_mae"] if f"{name}_mae" in res['metrics'] else res['metrics'][list(res['metrics'].keys())[3]],
        })
    summary_data.append({
        'Model': 'Ensemble (Calibrated)',
        'Pearson (r)': ens_metrics['Ensemble_Calibrated_pearson'],
        'RMSE': ens_metrics['Ensemble_Calibrated_rmse'],
        'Spearman (rho)': ens_metrics['Ensemble_Calibrated_spearman'],
        'MAE': ens_metrics['Ensemble_Calibrated_mae'],
    })
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(fig_dir, f"{exp_name}_metrics_benchmark.csv"), index=False)
    print("\nBenchmark Summary Table:")
    print(summary_df.to_string(index=False))

    return summary_df, sub_ensemble_path


if __name__ == "__main__":
    feat_dir = r"C:\Users\Somya\.gemini\antigravity\scratch\shl_grammar_scoring\data\features"
    train_path = os.path.join(feat_dir, "audio_features_train.csv")
    test_path = os.path.join(feat_dir, "audio_features_test.csv")
    run_experiment_pipeline(train_path, test_path, exp_name="v1_acoustic_baseline")
