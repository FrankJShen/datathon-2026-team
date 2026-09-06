import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from scipy.special import logit, expit
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
TARGET_COL = "default"
ID_COL = "client_id"

N_SPLITS = 10
RANDOM_STATE = 2024
CONFIDENCE_HIGH = 0.95
CONFIDENCE_LOW = 0.05
KNOWN_BASE_RATE = 0.2212

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    if "EDUCATION" in df.columns:
        df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4}).astype('category')
        df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3}).astype('category')
        df["SEX"] = df["SEX"].astype('category')

    for i in range(1, 7):
        df[f"UTIL_{i}"] = df[f"BILL_AMT{i}"] / df["LIMIT_BAL"].replace(0, np.nan)
    df['UTIL_AVG'] = df[[f'UTIL_{i}' for i in range(1, 7)]].mean(axis=1)

    for i in range(1, 6):
        df[f"PAY_RATIO_{i}"] = df[f"PAY_AMT{i}"] / df[f"BILL_AMT{i+1}"].clip(lower=0).replace(0, np.nan)
    df['PAYRATIO_AVG'] = df[[f'PAY_RATIO_{i}' for i in range(1, 6)]].mean(axis=1)

    pay_status_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    df["MAX_DELAY"] = df[pay_status_cols].max(axis=1)
    df["SUM_DELAYS"] = df[pay_status_cols].clip(lower=0).sum(axis=1)
    df['NUM_MONTHS_LATE'] = (df[pay_status_cols] > 0).sum(axis=1)
    df['WORSENING'] = df['PAY_0'] - df['PAY_6']

    bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
    pay_cols = [f'PAY_AMT{i}' for i in range(1, 7)]
    
    df['BILL_TREND'] = df['BILL_AMT1'] - df['BILL_AMT6']
    df['HEADROOM'] = df['LIMIT_BAL'] - df['BILL_AMT1']
    df['BILL_STD'] = df[bill_cols].std(axis=1)
    df['PAYAMT_STD'] = df[pay_cols].std(axis=1)

    # Allow tree algorithms to handle NaNs natively; do not fill with 0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def align_base_rate(probs, target_mean):
    safe_probs = np.clip(probs, 1e-15, 1 - 1e-15)
    logits = logit(safe_probs)
    
    def objective(delta):
        shifted_probs = expit(logits + delta)
        return np.mean(shifted_probs) - target_mean
    
    optimal_delta = brentq(objective, -10.0, 10.0)
    return expit(logits + optimal_delta)

def main():
    print("[*] Loading and engineering datasets...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    full_train = engineer_features(train_df)
    full_test = engineer_features(test_df)
    
    feature_cols = [c for c in full_train.columns if c not in [ID_COL, TARGET_COL]]
    
    X_orig = full_train[feature_cols]
    y_orig = full_train[TARGET_COL].values
    X_test = full_test[feature_cols]

    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.036019,
        "num_leaves": 51,
        "max_depth": 5,
        "min_child_weight": 6.709387,
        "subsample": 0.664436,
        "colsample_bytree": 0.957112,
        "reg_alpha": 0.053511,
        "reg_lambda": 0.314483,
        "n_estimators": 700, # Constrained to prevent overfitting
        "n_jobs": -1,
        "verbose": -1,
        "random_state": RANDOM_STATE
    }

    print("[*] Phase 1: Generating Pseudo-Labels on Test Set...")
    pseudo_model = lgb.LGBMClassifier(**lgb_params)
    pseudo_model.fit(X_orig, y_orig)
    raw_test_preds = pseudo_model.predict_proba(X_test)[:, 1]

    high_conf_idx = np.where(raw_test_preds > CONFIDENCE_HIGH)[0]
    low_conf_idx = np.where(raw_test_preds < CONFIDENCE_LOW)[0]

    pseudo_X_high = X_test.iloc[high_conf_idx].copy()
    pseudo_y_high = np.ones(len(high_conf_idx))
    pseudo_X_low = X_test.iloc[low_conf_idx].copy()
    pseudo_y_low = np.zeros(len(low_conf_idx))

    X_aug = pd.concat([X_orig, pseudo_X_high, pseudo_X_low], ignore_index=True)
    y_aug = np.concatenate([y_orig, pseudo_y_high, pseudo_y_low])

    augmented_df = X_aug.copy()
    augmented_df[TARGET_COL] = y_aug
    augmented_filename = "augmented_train.csv"
    augmented_df.to_csv(augmented_filename, index=False)

    print(f"    Added {len(high_conf_idx)} positive and {len(low_conf_idx)} negative pseudo-labels.")
    print(f"    New Training Set Size: {len(X_aug)} (Original: {len(X_orig)})")

    print(f"\n[*] Phase 2: Training Calibrated LightGBM on Augmented Data ({N_SPLITS} Folds)...")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    oof_preds = np.zeros(len(X_aug))
    final_test_preds = np.zeros(len(X_test))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_aug, y_aug)):
        X_tr_full, y_tr_full = X_aug.iloc[tr_idx], y_aug[tr_idx]
        X_va, y_va = X_aug.iloc[va_idx], y_aug[va_idx]
        
        X_tr, X_calib, y_tr, y_calib = train_test_split(
            X_tr_full, y_tr_full, test_size=0.15, random_state=RANDOM_STATE + fold, stratify=y_tr_full
        )
        
        base_model = lgb.LGBMClassifier(**lgb_params)
        base_model.fit(
            X_tr, y_tr, 
            eval_set=[(X_va, y_va)], 
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        calibrated_model = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv='prefit')
        calibrated_model.fit(X_calib, y_calib)
        
        val_fold_preds = calibrated_model.predict_proba(X_va)[:, 1]
        oof_preds[va_idx] = val_fold_preds
        final_test_preds += calibrated_model.predict_proba(X_test)[:, 1] / N_SPLITS
        
        print(f"    Fold {fold + 1:02d} Calibrated Log Loss: {log_loss(y_va, val_fold_preds):.5f}")

    overall_loss = log_loss(y_aug, oof_preds)
    print(f"\n[+] Augmented Out-of-Fold Log Loss: {overall_loss:.5f}")

    print("\n[*] Phase 3: Applying Log Loss Penalty Clipping and Base Rate Alignment...")
    clipped_test_preds = np.clip(final_test_preds, 0.015, 0.985)
    final_aligned_preds = align_base_rate(clipped_test_preds, KNOWN_BASE_RATE)

    submission = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: final_aligned_preds})
    filename = f"tuned_master_submission_{overall_loss:.5f}.csv"
    submission.to_csv(filename, index=False)
    print(f"[+] Saved final submission to {filename}")

if __name__ == "__main__":
    main()