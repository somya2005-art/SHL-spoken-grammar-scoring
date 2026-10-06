import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from src.evaluation import compute_metrics, calibrate_predictions
import warnings
warnings.filterwarnings("ignore")

def get_stratified_folds(y, n_splits=5, random_state=42):
    """
    Creates stratified K-fold splits by binning continuous scores into integer intervals.
    """
    y_binned = np.round(y * 2).astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(np.zeros(len(y)), y_binned))


def train_lgb_kfold(X, y, X_test, folds, params=None):
    """
    Trains LightGBM Regressor with K-Fold cross validation.
    """
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'n_estimators': 800,
            'learning_rate': 0.03,
            'num_leaves': 15,
            'max_depth': 4,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_samples': 15,
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    feature_importances = np.zeros(X.shape[1])
    models = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        val_pred = model.predict(X_va)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(X_test) / len(folds)
        feature_importances += model.feature_importances_ / len(folds)
        models.append(model)

    metrics = compute_metrics(y, oof_preds, prefix="LGBM")
    return oof_preds, test_preds, metrics, feature_importances, models


def train_xgb_kfold(X, y, X_test, folds, params=None):
    """
    Trains XGBoost Regressor with K-Fold cross validation.
    """
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': 800,
            'learning_rate': 0.03,
            'max_depth': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1
        }

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    feature_importances = np.zeros(X.shape[1])
    models = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_val)],
            verbose=False
        )

        val_pred = model.predict(X_va)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(X_test) / len(folds)
        feature_importances += model.feature_importances_ / len(folds)
        models.append(model)

    metrics = compute_metrics(y, oof_preds, prefix="XGB")
    return oof_preds, test_preds, metrics, feature_importances, models


def train_catboost_kfold(X, y, X_test, folds, params=None):
    """
    Trains CatBoost Regressor with K-Fold cross validation.
    """
    if params is None:
        params = {
            'iterations': 800,
            'learning_rate': 0.03,
            'depth': 4,
            'l2_leaf_reg': 3.0,
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'random_seed': 42,
            'verbose': False
        }

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    feature_importances = np.zeros(X.shape[1])
    models = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = CatBoostRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_val),
            early_stopping_rounds=50,
            verbose=False
        )

        val_pred = model.predict(X_va)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(X_test) / len(folds)
        feature_importances += model.get_feature_importance() / len(folds)
        models.append(model)

    metrics = compute_metrics(y, oof_preds, prefix="CatBoost")
    return oof_preds, test_preds, metrics, feature_importances, models


def train_ridge_kfold(X, y, X_test, folds, alpha=10.0):
    """
    Trains Ridge Regressor with scaled features.
    """
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    models = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_val = X.iloc[val_idx], y.iloc[val_idx]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr.fillna(0))
        X_va_scaled = scaler.transform(X_va.fillna(0))
        X_te_scaled = scaler.transform(X_test.fillna(0))

        model = Ridge(alpha=alpha, random_state=42)
        model.fit(X_tr_scaled, y_tr)

        val_pred = model.predict(X_va_scaled)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(X_te_scaled) / len(folds)
        models.append((scaler, model))

    metrics = compute_metrics(y, oof_preds, prefix="Ridge")
    return oof_preds, test_preds, metrics, models


def optimize_ensemble_weights(y_true, oof_list):
    """
    Finds optimal weights for combining multiple out-of-fold predictions
    to maximize Pearson correlation and minimize RMSE.
    """
    n_models = len(oof_list)
    oof_matrix = np.column_stack(oof_list)

    def loss_func(weights):
        weights = weights / np.sum(weights)
        blend = np.dot(oof_matrix, weights)
        corr, _ = compute_metrics(y_true, blend)['pearson'], 0
        rmse = compute_metrics(y_true, blend)['rmse']
        # Composite loss: -pearson + 0.5 * rmse
        return -corr + 0.3 * rmse

    init_weights = np.ones(n_models) / n_models
    bounds = [(0, 1) for _ in range(n_models)]
    constraints = ({'type': 'eq', 'fun': lambda w: 1 - np.sum(w)})

    res = minimize(loss_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_weights = res.x / np.sum(res.x)
    return optimal_weights
