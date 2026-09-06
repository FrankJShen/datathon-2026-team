# ============================================================
# IMPORT
# ============================================================

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

import xgboost as xgb

# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "default"
ID_COL = "client_id"

RANDOM_STATE = 42
N_SPLITS = 5

SEEDS = [42, 123, 2026, 777, 31415]

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = (
    PROJECT_ROOT /
    "inter-uni-datathon-stream-1-credit-card-clients"
)

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

SUBMISSION_DIR = PROJECT_ROOT / "submissions"
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)


if not TRAIN_PATH.exists():
    raise FileNotFoundError(
        f"Could not find training data at: {TRAIN_PATH}"
    )

if not TEST_PATH.exists():
    raise FileNotFoundError(
        f"Could not find test data at: {TEST_PATH}"
    )


print("Project root:", PROJECT_ROOT)
print("Data directory:", DATA_DIR)
print("Train path:", TRAIN_PATH)
print("Test path:", TEST_PATH)
print("Submission directory:", SUBMISSION_DIR)

# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("Train shape:", train.shape)
print("Test shape:", test.shape)

print("\nTraining data preview:")
print(train.head())

# ============================================================
# DATA CHECKS
# ============================================================

print("Train rows:", len(train))
print("Test rows:", len(test))

print("\nTarget present in train:", TARGET in train.columns)
print("Target present in test:", TARGET in test.columns)

print("\nDuplicate train client IDs:", train[ID_COL].duplicated().sum())
print("Duplicate test client IDs:", test[ID_COL].duplicated().sum())

print("\nMissing values in train:", train.isna().sum().sum())
print("Missing values in test:", test.isna().sum().sum())

print("\nTarget counts:")
print(train[TARGET].value_counts().sort_index())

print("\nTarget proportions:")
print(
    train[TARGET]
    .value_counts(normalize=True)
    .sort_index()
)

print("\nColumns:")
print(train.columns.tolist())

assert TARGET in train.columns
assert TARGET not in test.columns

assert ID_COL in train.columns
assert ID_COL in test.columns

assert train[ID_COL].is_unique
assert test[ID_COL].is_unique

print("\nBasic data checks passed.")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    df = df.copy()

    PAY_COLS = [
        "PAY_0", "PAY_2", "PAY_3",
        "PAY_4", "PAY_5", "PAY_6"
    ]

    BILL_COLS = [
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
        "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"
    ]

    PAYAMT_COLS = [
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
        "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]

    # Credit utilisation
    for i in range(1, 7):
        df[f"UTIL{i}"] = (
            df[f"BILL_AMT{i}"] /
            df["LIMIT_BAL"].replace(0, np.nan)
        )

    df["UTIL_AVG"] = df[
        [f"UTIL{i}" for i in range(1, 7)]
    ].mean(axis=1)

    # Payment-to-bill ratios
    for i in range(1, 6):
        bill = df[f"BILL_AMT{i+1}"]
        pay = df[f"PAY_AMT{i}"]

        df[f"PAYRATIO{i}"] = np.where(
            bill > 0,
            pay / bill,
            np.where(pay > 0, 1.0, np.nan)
        )

    df["PAYRATIO_AVG"] = df[
        [f"PAYRATIO{i}" for i in range(1, 6)]
    ].mean(axis=1)

    # Repayment behaviour
    df["PAY_MAX"] = df[PAY_COLS].max(axis=1)
    df["PAY_MEAN"] = df[PAY_COLS].mean(axis=1)

    df["PAY_SUM_LATE"] = (
        df[PAY_COLS]
        .clip(lower=0)
        .sum(axis=1)
    )

    df["NUM_MONTHS_LATE"] = (
        df[PAY_COLS] > 0
    ).sum(axis=1)

    df["WORSENING"] = (
        df["PAY_0"] -
        df["PAY_6"]
    )

    # Bill/payment summaries
    df["BILL_TREND"] = (
        df["BILL_AMT1"] -
        df["BILL_AMT6"]
    )

    df["BILL_AVG"] = df[BILL_COLS].mean(axis=1)
    df["PAYAMT_AVG"] = df[PAYAMT_COLS].mean(axis=1)

    # Category cleaning
    df["EDUCATION"] = df["EDUCATION"].replace({
        0: 4,
        5: 4,
        6: 4
    })

    df["MARRIAGE"] = df["MARRIAGE"].replace({
        0: 3
    })

    # Headroom and volatility
    df["HEADROOM"] = (
        df["LIMIT_BAL"] -
        df["BILL_AMT1"]
    )

    df["BILL_STD"] = df[BILL_COLS].std(axis=1)
    df["PAYAMT_STD"] = df[PAYAMT_COLS].std(axis=1)

    # Remove infinities created by ratios
    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    return df

train_fe = engineer_features(train)
test_fe = engineer_features(test)

FEATURES = [
    c for c in train_fe.columns
    if c not in [TARGET, ID_COL]
]

X = train_fe[FEATURES].fillna(0)
y = train_fe[TARGET].copy()

X_test = test_fe[FEATURES].fillna(0)

print("Training matrix:", X.shape)
print("Test matrix:", X_test.shape)
print("Number of features:", len(FEATURES))

assert list(X.columns) == list(X_test.columns)
assert X.shape[1] == X_test.shape[1]

print("\nFeature generation passed.")

# ============================================================
# BASELINE XGBOOST CONFIGURATION
# ============================================================

XGB_PARAMS = {
    "n_estimators": 600,
    "max_depth": 6,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "eval_metric": "logloss",
    "n_jobs": -1,
    "tree_method": "hist"
}


def make_xgb(seed):
    return xgb.XGBClassifier(
        **XGB_PARAMS,
        random_state=seed
    )


print(XGB_PARAMS)

# ============================================================
# 5-FOLD STRATIFIED VALIDATION
# ============================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

oof_pred = np.zeros(len(train))
fold_losses = []

for fold, (train_idx, valid_idx) in enumerate(
    skf.split(X, y),
    start=1
):

    print(f"Training fold {fold}...")

    X_train_fold = X.iloc[train_idx]
    X_valid_fold = X.iloc[valid_idx]

    y_train_fold = y.iloc[train_idx]
    y_valid_fold = y.iloc[valid_idx]

    model = make_xgb(seed=RANDOM_STATE)

    model.fit(
        X_train_fold,
        y_train_fold
    )

    valid_pred = model.predict_proba(
        X_valid_fold
    )[:, 1]

    oof_pred[valid_idx] = valid_pred

    fold_loss = log_loss(
        y_valid_fold,
        valid_pred
    )

    fold_losses.append(fold_loss)

    print(
        f"Fold {fold} log loss: "
        f"{fold_loss:.6f}"
    )


cv_loss = log_loss(y, oof_pred)

print("\n============================")
print("VALIDATION RESULTS")
print("============================")

for i, score in enumerate(fold_losses, start=1):
    print(f"Fold {i}: {score:.6f}")

print("\nMean fold log loss:", np.mean(fold_losses))
print("OOF log loss:", cv_loss)

# ============================================================
# FULL-DATA SINGLE MODEL
# ============================================================

baseline_model = make_xgb(seed=42)

baseline_model.fit(
    X,
    y
)

baseline_test_pred = baseline_model.predict_proba(
    X_test
)[:, 1]

print("Predictions:", len(baseline_test_pred))
print("Mean prediction:", baseline_test_pred.mean())
print("Minimum:", baseline_test_pred.min())
print("Maximum:", baseline_test_pred.max())


# ============================================================
# FIVE-SEED XGBOOST ENSEMBLE
# ============================================================

seed_predictions = []

for seed in SEEDS:

    print(f"Training seed {seed}...")

    model = make_xgb(seed)

    model.fit(
        X,
        y
    )

    pred = model.predict_proba(
        X_test
    )[:, 1]

    seed_predictions.append(pred)


seed_predictions = np.array(seed_predictions)

ensemble_pred = seed_predictions.mean(axis=0)

print("\nFinished.")

print(
    "Seed prediction matrix:",
    seed_predictions.shape
)

print(
    "Final prediction shape:",
    ensemble_pred.shape
)

print(
    "\nMean probability:",
    ensemble_pred.mean()
)

print(
    "Minimum probability:",
    ensemble_pred.min()
)

print(
    "Maximum probability:",
    ensemble_pred.max()
)


# ============================================================
# SUBMISSION GENERATION
# ============================================================

submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    TARGET: ensemble_pred
})

OUTPUT_PATH = (
    SUBMISSION_DIR /
    "submission_mani_test_1.csv"
)

submission.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSubmission preview:")
print(submission.head())

print("\nRows:", len(submission))

print(
    "Missing:",
    submission[TARGET].isna().sum()
)

print(
    "Mean:",
    submission[TARGET].mean()
)

print("\nSaved to:")
print(OUTPUT_PATH.resolve())

