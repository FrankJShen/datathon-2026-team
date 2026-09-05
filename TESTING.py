"""
Credit Default Prediction — Ensemble Pipeline
================================================
Metric: binary log loss (rewards calibrated probabilities, punishes confident
mistakes hard). Everything here is chosen with that in mind.

Design:
  1. Feature engineering: utilization, payment ratios, delinquency counts,
     AND trend/volatility features (is the customer getting worse or better
     over the 6-month window?) — vectorized, no per-row Python loops.
  2. Three different gradient-boosting families (LightGBM, XGBoost, CatBoost).
     They make different mistakes on different customers, so blending them
     usually beats any single one on log loss.
  3. Each fold: a 3-way split (fit / calibrate / validate) so the validation
     fold is touched exactly once, only for scoring — no calibration leakage.
  4. Blend weights across the 3 models are chosen by directly minimizing
     out-of-fold log loss (not just averaged equally).
  5. A final calibration check on the blended OOF predictions: try raw /
     isotonic / Platt, keep whichever actually lowers log loss. If none help,
     none is applied — this never blindly makes things worse.

Honesty check: no code is "unbeatable". This is the strongest well-validated
approach for this problem type; the rest comes down to your specific data,
feature ideas, and time left on the clock.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

try:
    from sklearn.frozen import FrozenEstimator
    HAS_FROZEN_ESTIMATOR = True
except ImportError:
    HAS_FROZEN_ESTIMATOR = False

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
POSSIBLE_PATHS = [
    Path("inter-uni-datathon-stream-1-credit-card-clients"),
    Path("data/raw"),
    Path("."),
]

TARGET_COL = "default"
ID_COL = "client_id"
N_SPLITS = 5
N_REPEATS = 1          # bump to 2-3 for more stable OOF estimates if you have spare time
CALIB_FRACTION = 0.15  # slice of each fold's training data reserved for calibration
RANDOM_STATE = 42
SUBMISSION_DIR = Path("submissions")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

MODEL_TYPES = ["lgb", "xgb", "cat"]


def locate_data():
    for p in POSSIBLE_PATHS:
        if (p / "train.csv").exists() and (p / "test.csv").exists():
            print(f"[*] Found datasets in: {p.resolve()}")
            return p / "train.csv", p / "test.csv"
    raise FileNotFoundError("Could not locate train.csv and test.csv.")


# --------------------------------------------------------------------------
# Feature engineering — all vectorized (matrix ops), no per-row Python loops
# --------------------------------------------------------------------------
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]   # most recent first
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]  # most recent first
PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]  # most recent first
CAT_COLS = ["SEX", "EDUCATION", "MARRIAGE"]


def _trend_slope(df, cols):
    """Vectorized linear-regression slope across the 6 monthly columns.
    Columns must be ordered most-recent-first. Positive = getting worse/bigger
    recently; negative = improving. One matrix expression, no row loop."""
    Y = df[cols].to_numpy(dtype=float)
    x = np.arange(len(cols), 0, -1)  # e.g. [6,5,4,3,2,1]
    x_c = x - x.mean()
    y_c = Y - Y.mean(axis=1, keepdims=True)
    return (y_c * x_c).sum(axis=1) / (x_c ** 2).sum()


def _longest_late_streak(df, cols):
    """Longest run of consecutive late months. 6 columns -> 6 cheap vector ops,
    not a per-row loop."""
    late = (df[cols].to_numpy() > 0).astype(int)
    streak = np.zeros_like(late)
    streak[:, 0] = late[:, 0]
    for j in range(1, late.shape[1]):
        streak[:, j] = (streak[:, j - 1] + 1) * late[:, j]
    return streak.max(axis=1)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Utilization: bill vs credit limit, each month + summary stats
    for i in range(1, 7):
        df[f"UTIL_{i}"] = df[f"BILL_AMT{i}"] / (df["LIMIT_BAL"] + 1.0)
    util_cols = [f"UTIL_{i}" for i in range(1, 7)]
    df["UTIL_MEAN"] = df[util_cols].mean(axis=1)
    df["UTIL_MAX"] = df[util_cols].max(axis=1)

    # Payment-to-bill ratio: this month's payment vs. the bill it paid off
    for i in range(1, 6):
        df[f"PAY_RATIO_{i}"] = df[f"PAY_AMT{i}"] / (df[f"BILL_AMT{i+1}"].clip(lower=0) + 1.0)
    df["PAY_RATIO_MEAN"] = df[[f"PAY_RATIO_{i}" for i in range(1, 6)]].mean(axis=1)

    # Recent deltas
    df["BILL_AMT_DIFF1"] = df["BILL_AMT1"] - df["BILL_AMT2"]
    df["PAY_AMT_DIFF1"] = df["PAY_AMT1"] - df["PAY_AMT2"]

    # 6-month aggregates
    df["BILL_MEAN"] = df[BILL_COLS].mean(axis=1)
    df["BILL_MAX"] = df[BILL_COLS].max(axis=1)
    df["BILL_STD"] = df[BILL_COLS].std(axis=1)
    df["PAY_SUM"] = df[PAYAMT_COLS].sum(axis=1)
    df["PAY_MEAN"] = df[PAYAMT_COLS].mean(axis=1)
    df["PAY_STD"] = df[PAYAMT_COLS].std(axis=1)

    # Delinquency
    df["MAX_DELAY"] = df[PAY_STATUS_COLS].max(axis=1)
    df["NUM_DELAYS"] = (df[PAY_STATUS_COLS] > 0).sum(axis=1)
    df["SUM_DELAYS"] = df[PAY_STATUS_COLS].clip(lower=0).sum(axis=1)
    df["EVER_SERIOUS_DELINQ"] = (df["MAX_DELAY"] >= 2).astype(int)
    df["MAX_LATE_STREAK"] = _longest_late_streak(df, PAY_STATUS_COLS)

    # Trend features — is the customer getting worse or better over 6 months?
    df["BILL_TREND"] = _trend_slope(df, BILL_COLS)
    df["PAY_TREND"] = _trend_slope(df, PAYAMT_COLS)
    df["DELINQ_TREND"] = _trend_slope(df, PAY_STATUS_COLS)

    # Simple interaction
    df["CREDIT_TO_AGE"] = df["LIMIT_BAL"] / (df["AGE"] + 1.0)

    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    return df


# --------------------------------------------------------------------------
# Model builders — one fresh, unfit model per fold/model-type
# --------------------------------------------------------------------------
def build_model(model_type, scale_pos_weight, seed):
    if model_type == "lgb":
        return lgb.LGBMClassifier(
            objective="binary", metric="binary_logloss",
            n_estimators=2000, learning_rate=0.02,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, min_child_samples=30,
            scale_pos_weight=scale_pos_weight,
            random_state=seed, n_jobs=-1, verbose=-1,
        )
    if model_type == "xgb":
        return xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            n_estimators=2000, learning_rate=0.02, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
            scale_pos_weight=scale_pos_weight,
            random_state=seed, n_jobs=-1, tree_method="hist",
            early_stopping_rounds=50, verbosity=0,
        )
    if model_type == "cat":
        return CatBoostClassifier(
            iterations=2000, learning_rate=0.02, depth=6, l2_leaf_reg=3.0,
            loss_function="Logloss", eval_metric="Logloss",
            scale_pos_weight=scale_pos_weight, cat_features=CAT_COLS,
            random_seed=seed, early_stopping_rounds=50, verbose=False,
        )
    raise ValueError(model_type)


def fit_fold_model(model_type, X_fit, y_fit, X_calib, y_calib, X_va, X_test, seed):
    """Fit one model on the fit-slice, calibrate on the calib-slice (never the
    validation fold), then predict on val + test. X_va is touched exactly once,
    here, purely for scoring."""
    spw = (y_fit == 0).sum() / max((y_fit == 1).sum(), 1)
    model = build_model(model_type, spw, seed)

    if model_type == "lgb":
        model.fit(X_fit, y_fit, eval_X=X_calib, eval_y=y_calib,
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    elif model_type == "xgb":
        model.fit(X_fit, y_fit, eval_set=[(X_calib, y_calib)], verbose=False)
    elif model_type == "cat":
        model.fit(X_fit, y_fit, eval_set=(X_calib, y_calib), use_best_model=True, verbose=False)

    calib_input = FrozenEstimator(model) if HAS_FROZEN_ESTIMATOR else model
    cv_arg = None if HAS_FROZEN_ESTIMATOR else "prefit"
    calibrated = CalibratedClassifierCV(estimator=calib_input, method="sigmoid", cv=cv_arg)
    calibrated.fit(X_calib, y_calib)

    val_pred = calibrated.predict_proba(X_va)[:, 1]
    test_pred = calibrated.predict_proba(X_test)[:, 1]
    return val_pred, test_pred


# --------------------------------------------------------------------------
# Cross-validated training for all 3 model types
# --------------------------------------------------------------------------
def train_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame):
    print("[*] Engineering features...")
    full_train = engineer_features(train_df)
    full_test = engineer_features(test_df)

    feature_cols = [c for c in full_train.columns if c not in [ID_COL, TARGET_COL]]
    X = full_train[feature_cols]
    y = full_train[TARGET_COL].to_numpy()
    X_test = full_test[feature_cols]
    print(f"[*] {len(feature_cols)} features, {len(X)} training rows, "
          f"default rate = {y.mean():.3f}")

    oof = {m: np.zeros(len(X)) for m in MODEL_TYPES}
    oof_count = np.zeros(len(X))
    test_pred_sum = {m: np.zeros(len(X_test)) for m in MODEL_TYPES}
    n_fold_fits = 0

    for repeat in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_va = X.iloc[val_idx]
            seed = RANDOM_STATE + repeat * 100 + fold

            # One fit/calibration split per fold, shared across all 3 model types —
            # keeps the comparison fair and halves the bookkeeping.
            X_fit, X_calib, y_fit, y_calib = train_test_split(
                X_tr, y_tr, test_size=CALIB_FRACTION, stratify=y_tr, random_state=seed
            )

            for model_type in MODEL_TYPES:
                val_pred, test_pred = fit_fold_model(
                    model_type, X_fit, y_fit, X_calib, y_calib, X_va, X_test, seed
                )
                val_pred = np.clip(val_pred, 1e-15, 1 - 1e-15)
                test_pred = np.clip(test_pred, 1e-15, 1 - 1e-15)
                oof[model_type][val_idx] += val_pred
                test_pred_sum[model_type] += test_pred

            oof_count[val_idx] += 1
            n_fold_fits += 1
            print(f"    repeat {repeat+1}/{N_REPEATS}  fold {fold+1}/{N_SPLITS} done")

    for model_type in MODEL_TYPES:
        oof[model_type] /= oof_count
        test_pred_sum[model_type] /= n_fold_fits
        score = log_loss(y, oof[model_type])
        print(f"[*] {model_type.upper():4s} OOF log loss: {score:.5f}")

    # ----------------------------------------------------------------------
    # Blend weights: minimize OOF log loss directly (not just equal-weight avg)
    # ----------------------------------------------------------------------
    oof_matrix = np.column_stack([oof[m] for m in MODEL_TYPES])
    test_matrix = np.column_stack([test_pred_sum[m] for m in MODEL_TYPES])

    def blend_loss(w):
        blend = np.clip(oof_matrix @ w, 1e-15, 1 - 1e-15)
        return log_loss(y, blend)

    n_models = len(MODEL_TYPES)
    result = minimize(
        blend_loss, x0=np.full(n_models, 1 / n_models),
        method="SLSQP", bounds=[(0, 1)] * n_models,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
    )
    weights = result.x
    print(f"[*] Blend weights ({', '.join(MODEL_TYPES)}): "
          f"{', '.join(f'{w:.3f}' for w in weights)}")

    oof_blend = np.clip(oof_matrix @ weights, 1e-15, 1 - 1e-15)
    test_blend = np.clip(test_matrix @ weights, 1e-15, 1 - 1e-15)
    blend_score = log_loss(y, oof_blend)
    print(f"[+] Blended OOF log loss (raw): {blend_score:.5f}")

    # ----------------------------------------------------------------------
    # Final calibration check: only keep it if it actually helps OOF log loss
    # ----------------------------------------------------------------------
    candidates = {"none": (oof_blend, test_blend)}

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_blend, y)
    candidates["isotonic"] = (
        np.clip(iso.predict(oof_blend), 1e-15, 1 - 1e-15),
        np.clip(iso.predict(test_blend), 1e-15, 1 - 1e-15),
    )

    platt = LogisticRegression()
    platt.fit(oof_blend.reshape(-1, 1), y)
    candidates["platt"] = (
        np.clip(platt.predict_proba(oof_blend.reshape(-1, 1))[:, 1], 1e-15, 1 - 1e-15),
        np.clip(platt.predict_proba(test_blend.reshape(-1, 1))[:, 1], 1e-15, 1 - 1e-15),
    )

    scored = {name: log_loss(y, oof_c) for name, (oof_c, _) in candidates.items()}
    best_name = min(scored, key=scored.get)
    for name, s in scored.items():
        flag = "  <- selected" if name == best_name else ""
        print(f"    calibration '{name}': OOF log loss {s:.5f}{flag}")

    final_test_pred = candidates[best_name][1]
    final_score = scored[best_name]
    print(f"\n[+] Final OOF log loss: {final_score:.5f}")

    return full_test[ID_COL], final_test_pred, final_score


def export_submission(ids, predictions, score):
    submission = pd.DataFrame({ID_COL: ids, TARGET_COL: predictions})
    filename = SUBMISSION_DIR / f"submission_logloss_{score:.5f}.csv"
    submission.to_csv(filename, index=False)
    print(f"[+] Submission written to: {filename}")
    print(submission.head())


def main():
    train_path, test_path = locate_data()
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    ids, predictions, score = train_and_evaluate(train_df, test_df)
    export_submission(ids, predictions, score)


if __name__ == "__main__":
    main()