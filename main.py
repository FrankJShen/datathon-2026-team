from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
import lightgbm as lgb

# --- Configuration & Paths ---
DATA_DIR = Path("data/raw")
SUBMISSION_DIR = Path("submissions")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target"
ID_COL = "id"
RANDOM_STATE = 42
N_SPLITS = 5


def load_datasets():
    """Load raw train and test datasets."""
    print("[*] Loading datasets...")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    return train, test, sample_sub


def feature_engineering(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Preprocess data, handle missing values, and extract features."""
    print("[*] Engineering features...")
    
    # Example: Simple numerical column processing
    feature_cols = [c for c in train_df.columns if c not in [TARGET_COL, ID_COL]]
    
    X = train_df[feature_cols].copy()
    y = train_df[TARGET_COL].copy()
    X_test = test_df[feature_cols].copy()

    # Fill missing values (baseline)
    X = X.fillna(X.median(numeric_only=True))
    X_test = X_test.fillna(X.median(numeric_only=True))

    return X, y, X_test, feature_cols


def train_cross_validation(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame):
    """Train models using K-Fold cross validation and generate out-of-fold predictions."""
    print(f"[*] Running {N_SPLITS}-Fold Cross Validation...")
    
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    oof_predictions = np.zeros(len(X))
    test_predictions = np.zeros(len(X_test))
    models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        val_preds = model.predict(X_val)
        oof_predictions[val_idx] = val_preds
        test_predictions += model.predict(X_test) / N_SPLITS
        models.append(model)

        fold_score = np.sqrt(mean_squared_error(y_val, val_preds))
        print(f"    Fold {fold + 1} RMSE: {fold_score:.5f}")

    cv_score = np.sqrt(mean_squared_error(y, oof_predictions))
    print(f"[+] Overall CV RMSE: {cv_score:.5f}")

    return test_predictions, cv_score


def generate_submission(sample_sub: pd.DataFrame, test_predictions: np.ndarray, cv_score: float):
    """Save test predictions to a submission file tagged with the validation score."""
    sample_sub[TARGET_COL] = test_predictions
    output_filename = SUBMISSION_DIR / f"sub_cv_{cv_score:.4f}.csv"
    sample_sub.to_csv(output_filename, index=False)
    print(f"[+] Submission saved to: {output_filename}")


def main():
    train_df, test_df, sample_sub = load_datasets()
    X, y, X_test, features = feature_engineering(train_df, test_df)
    test_preds, cv_score = train_cross_validation(X, y, X_test)
    generate_submission(sample_sub, test_preds, cv_score)


if __name__ == "__main__":
    main()