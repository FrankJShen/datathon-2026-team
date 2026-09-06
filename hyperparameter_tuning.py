import pandas as pd
import numpy as np
import optuna
import warnings
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Hide warning text so the terminal output remains clean
warnings.filterwarnings("ignore")
# Tell Optuna to only print the important progress updates, not every tiny detail
optuna.logging.set_verbosity(optuna.logging.INFO)

# File locations and constants
TRAIN_PATH = "train.csv"
TARGET_COL = "default"
ID_COL = "client_id"
N_SPLITS = 10        # 10-fold cross validation to match the final pipeline
RANDOM_STATE = 2024  # Locked seed so the data splits identically every time

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies the exact same math and logic as the final submission model."""
    df = df.copy()
    
    # Group rare categorical values together to prevent the algorithm from memorizing random noise
    if "EDUCATION" in df.columns:
        df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4}).astype('category')
        df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3}).astype('category')
        df["SEX"] = df["SEX"].astype('category')

    # Utilisation: What percentage of their total credit limit are they actively using?
    for i in range(1, 7):
        df[f"UTIL_{i}"] = df[f"BILL_AMT{i}"] / df["LIMIT_BAL"].replace(0, np.nan)
    df['UTIL_AVG'] = df[[f'UTIL_{i}' for i in range(1, 7)]].mean(axis=1)

    # Repayment Ratio: What percentage of last month's bill did they actually pay off?
    for i in range(1, 6):
        df[f"PAY_RATIO_{i}"] = df[f"PAY_AMT{i}"] / df[f"BILL_AMT{i+1}"].clip(lower=0).replace(0, np.nan)
    df['PAYRATIO_AVG'] = df[[f'PAY_RATIO_{i}' for i in range(1, 6)]].mean(axis=1)

    # Delinquency: Track their worst late payment and if their behavior is accelerating downward
    pay_status_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    df["MAX_DELAY"] = df[pay_status_cols].max(axis=1)
    df["SUM_DELAYS"] = df[pay_status_cols].clip(lower=0).sum(axis=1)
    df['NUM_MONTHS_LATE'] = (df[pay_status_cols] > 0).sum(axis=1)
    df['WORSENING'] = df['PAY_0'] - df['PAY_6']

    # Volatility: Calculate standard deviation to see how erratic their spending/payments are
    bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
    pay_cols = [f'PAY_AMT{i}' for i in range(1, 7)]
    df['BILL_TREND'] = df['BILL_AMT1'] - df['BILL_AMT6']
    df['HEADROOM'] = df['LIMIT_BAL'] - df['BILL_AMT1']
    df['BILL_STD'] = df[bill_cols].std(axis=1)
    df['PAYAMT_STD'] = df[pay_cols].std(axis=1)

    # Do NOT fill missing values with 0. Let the tree algorithm route NaNs naturally.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def objective(trial, X, y):
    """The core scoring function that Optuna tries to minimize."""
    
    # Optuna dynamically guesses values within these specific ranges for every new trial
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 700, # Hard limit on trees to prevent overfitting to the public leaderboard
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
        "n_jobs": 8, # Locked to 8 threads to perfectly mimic the winning laptop hardware
        "verbose": -1,
        "random_state": RANDOM_STATE
    }

    # 10-Fold CV setup to evaluate how good the AI's parameter guesses are
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X))
    
    for train_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]
        
        # Build and train the LightGBM model with the current trial's guessed parameters
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)] # Stop training early if validation stops improving
        )
        # Record the predictions for the holdout fold to calculate the true score
        oof_preds[val_idx] = model.predict_proba(X_va)[:, 1]
        
    # Return the overall log loss score back to Optuna so it can learn and guess better next time
    return log_loss(y, oof_preds)

def main():
    print("[*] Loading and engineering datasets...")
    train_df = pd.read_csv(TRAIN_PATH)
    full_train = engineer_features(train_df)
    
    # Separate the clues (features) from the answers (targets)
    feature_cols = [c for c in full_train.columns if c not in [ID_COL, TARGET_COL]]
    X = full_train[feature_cols]
    y = full_train[TARGET_COL].values

    # Create the Optuna study that actively hunts for the lowest possible log loss score
    study = optuna.create_study(direction="minimize", study_name="Final_LGBM_Tune")
    # Run 50 different experimental trials
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50)
        
    print("\n[*] BEST HYPERPARAMETERS FOUND:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print(f"[*] Best OOF Log Loss: {study.best_value:.5f}")

if __name__ == "__main__":
    main()