from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

# Try importing LightGBM; fallback to sklearn's HistGradientBoosting if not installed
try:
    import lightgbm as lgb
    USE_LIGHTGBM = True
except ImportError:
    USE_LIGHTGBM = False

# --- Configuration & File Paths ---
POSSIBLE_PATHS = [
    Path("inter-uni-datathon-stream-1-credit-card-clients"),
    Path("data/raw"),
    Path(".")
]

TARGET_COL = "default"
ID_COL = "client_id"
N_SPLITS = 5
RANDOM_STATE = 42
SUBMISSION_DIR = Path("submissions")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)


def locate_data():
    """Find train.csv and test.csv in project folder variants."""
    for p in POSSIBLE_PATHS:
        if (p / "train.csv").exists() and (p / "test.csv").exists():
            print(f"[*] Found datasets in: {p.resolve()}")
            return p / "train.csv", p / "test.csv"
    raise FileNotFoundError("Could not locate train.csv and test.csv.")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create credit risk features: utilization, payment ratios, and delinquency stats."""
    df = df.copy()

    # 1. Credit Utilization Ratios (Bill / Credit Limit)
    for i in range(1, 7):
        df[f"UTIL_{i}"] = df[f"BILL_AMT{i}"] / (df["LIMIT_BAL"] + 1.0)

    # 2. Payment-to-Bill Ratios (Paid amount vs previous bill statement)
    for i in range(1, 6):
        df[f"PAY_RATIO_{i}"] = df[f"PAY_AMT{i}"] / (df[f"BILL_AMT{i+1}"].clip(lower=0) + 1.0)

    # 3. Trends and deltas (recent change in debt and repayment)
    df["BILL_AMT_DIFF1"] = df["BILL_AMT1"] - df["BILL_AMT2"]
    df["PAY_AMT_DIFF1"] = df["PAY_AMT1"] - df["PAY_AMT2"]

    # 4. Aggregations across the 6-month observation window
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    pay_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    pay_status_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

    df["BILL_MEAN"] = df[bill_cols].mean(axis=1)
    df["BILL_MAX"] = df[bill_cols].max(axis=1)
    df["PAY_SUM"] = df[pay_cols].sum(axis=1)
    df["PAY_MEAN"] = df[pay_cols].mean(axis=1)

    # 5. Delinquency indicators
    df["MAX_DELAY"] = df[pay_status_cols].max(axis=1)
    df["NUM_DELAYS"] = (df[pay_status_cols] > 0).sum(axis=1)
    df["SUM_DELAYS"] = df[pay_status_cols].clip(lower=0).sum(axis=1)

    return df


def train_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """5-Fold Stratified CV with calibrated probability prediction."""
    print("[*] Engineering domain features...")
    full_train = engineer_features(train_df)
    full_test = engineer_features(test_df)

    feature_cols = [c for c in full_train.columns if c not in [ID_COL, TARGET_COL]]
    X = full_train[feature_cols]
    y = full_train[TARGET_COL].values
    X_test = full_test[feature_cols]

    print(f"[*] Training on {len(feature_cols)} features using 5-Fold Stratified CV...")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof_predictions = np.zeros(len(X))
    test_predictions = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        if USE_LIGHTGBM:
            base_model = lgb.LGBMClassifier(
                objective="binary",
                metric="binary_logloss",
                learning_rate=0.03,
                n_estimators=600,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE + fold,
                n_jobs=-1,
                verbose=-1
            )
            base_model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            # Calibrate probabilities using Platt scaling (sigmoid)
            calibrated_model = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv="prefit")
            calibrated_model.fit(X_va, y_va)
            val_preds = calibrated_model.predict_proba(X_va)[:, 1]
            test_preds = calibrated_model.predict_proba(X_test)[:, 1]
        else:
            base_model = HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.04,
                random_state=RANDOM_STATE + fold,
                early_stopping=True,
                validation_fraction=0.1
            )
            calibrated_model = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=3)
            calibrated_model.fit(X_tr, y_tr)
            val_preds = calibrated_model.predict_proba(X_va)[:, 1]
            test_preds = calibrated_model.predict_proba(X_test)[:, 1]

        # Prevent extreme values from blowing up log-loss
        val_preds = np.clip(val_preds, 1e-15, 1 - 1e-15)
        test_preds = np.clip(test_preds, 1e-15, 1 - 1e-15)

        oof_predictions[val_idx] = val_preds
        test_predictions += test_preds / N_SPLITS

        fold_loss = log_loss(y_va, val_preds)
        print(f"    Fold {fold + 1} Log Loss: {fold_loss:.5f}")

    overall_log_loss = log_loss(y, oof_predictions)
    print(f"\n[+] Overall Out-of-Fold CV Log Loss: {overall_log_loss:.5f}")

    return test_predictions, overall_log_loss


def export_submission(test_df: pd.DataFrame, predictions: np.ndarray, cv_score: float):
    """Generate competition submission CSV."""
    submission = pd.DataFrame({
        ID_COL: test_df[ID_COL],
        TARGET_COL: predictions
    })
    filename = SUBMISSION_DIR / f"submission_logloss_{cv_score:.5f}.csv"
    submission.to_csv(filename, index=False)
    print(f"[+] Submission file written to: {filename}")
    print(submission.head())


def main():
    train_path, test_path = locate_data()
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    predictions, cv_score = train_and_evaluate(train_df, test_df)
    export_submission(test_df, predictions, cv_score)


if __name__ == "__main__":
    main()