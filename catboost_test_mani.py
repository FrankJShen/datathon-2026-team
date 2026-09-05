import pandas as pd
import numpy as np

from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss


# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = "inter-uni-datathon-stream-1-credit-card-clients"

train = pd.read_csv(f"{DATA_PATH}/train.csv")
test = pd.read_csv(f"{DATA_PATH}/test.csv")

print("Train:", train.shape)
print("Test:", test.shape)


# ============================================================
# COLUMNS
# ============================================================

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


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer(df):
    df = df.copy()

    # Utilisation
    for i in range(1, 7):
        df[f"UTIL{i}"] = (
            df[f"BILL_AMT{i}"]
            / df["LIMIT_BAL"].replace(0, np.nan)
        )

    util_cols = [f"UTIL{i}" for i in range(1, 7)]

    df["UTIL_AVG"] = df[util_cols].mean(axis=1)
    df["UTIL_MAX"] = df[util_cols].max(axis=1)
    df["UTIL_STD"] = df[util_cols].std(axis=1)
    df["UTIL_LAST"] = df["UTIL1"]


    # Payment-to-bill ratios
    for i in range(1, 6):

        bill = df[f"BILL_AMT{i+1}"]
        pay = df[f"PAY_AMT{i}"]

        df[f"PAYRATIO{i}"] = np.where(
            bill > 0,
            pay / bill,
            np.where(pay > 0, 1.0, np.nan)
        )

    ratio_cols = [f"PAYRATIO{i}" for i in range(1, 6)]

    df["PAYRATIO_AVG"] = df[ratio_cols].mean(axis=1)


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

    df["NUM_SEVERE_LATE"] = (
        df[PAY_COLS] >= 2
    ).sum(axis=1)

    # Recent repayment behaviour
    df["RECENT_3_PAY_STATUS"] = (
        df[["PAY_0", "PAY_2", "PAY_3"]]
        .mean(axis=1)
    )

    df["OLD_3_PAY_STATUS"] = (
        df[["PAY_4", "PAY_5", "PAY_6"]]
        .mean(axis=1)
    )

    df["PAY_STATUS_CHANGE"] = (
        df["RECENT_3_PAY_STATUS"]
        - df["OLD_3_PAY_STATUS"]
    )


    # Bills
    df["BILL_AVG"] = df[BILL_COLS].mean(axis=1)
    df["BILL_MAX"] = df[BILL_COLS].max(axis=1)
    df["BILL_STD"] = df[BILL_COLS].std(axis=1)

    df["BILL_TREND"] = (
        df["BILL_AMT1"]
        - df["BILL_AMT6"]
    )


    # Payments
    df["PAYAMT_AVG"] = df[PAYAMT_COLS].mean(axis=1)
    df["PAYAMT_TOTAL"] = df[PAYAMT_COLS].sum(axis=1)
    df["PAYAMT_STD"] = df[PAYAMT_COLS].std(axis=1)

    df["ZERO_PAY_MONTHS"] = (
        df[PAYAMT_COLS] == 0
    ).sum(axis=1)


    # Clean infinities
    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    df.fillna(0, inplace=True)

    return df


train_fe = engineer(train)
test_fe = engineer(test)


# ============================================================
# PREPARE DATA
# ============================================================

features = [
    c for c in train_fe.columns
    if c not in ["client_id", "default"]
]

X = train_fe[features].copy()
y = train_fe["default"].copy()

X_test = test_fe[features].copy()


# Treat these as genuine categories
cat_cols = [
    "SEX",
    "EDUCATION",
    "MARRIAGE"
]

for col in cat_cols:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

cat_indices = [
    X.columns.get_loc(c)
    for c in cat_cols
]


print("Features:", len(features))


# ============================================================
# 5-FOLD CATBOOST
# ============================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

fold_scores = []


for fold, (tr_idx, val_idx) in enumerate(
    skf.split(X, y),
    start=1
):

    print(f"\n===== Fold {fold} =====")

    X_train = X.iloc[tr_idx]
    y_train = y.iloc[tr_idx]

    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]


    model = CatBoostClassifier(

        loss_function="Logloss",
        eval_metric="Logloss",

        iterations=2000,
        learning_rate=0.025,

        depth=5,

        l2_leaf_reg=8,

        random_strength=0.5,

        random_seed=42 + fold,

        allow_writing_files=False,

        verbose=False
    )


    model.fit(

        X_train,
        y_train,

        cat_features=cat_indices,

        eval_set=(X_val, y_val),

        early_stopping_rounds=100,

        verbose=200
    )


    val_pred = model.predict_proba(X_val)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]


    oof[val_idx] = val_pred

    test_preds += test_pred / 5


    fold_loss = log_loss(y_val, val_pred)

    fold_scores.append(fold_loss)


    print(
        f"Fold {fold} log loss: "
        f"{fold_loss:.6f}"
    )

    print(
        "Best iteration:",
        model.get_best_iteration()
    )


# ============================================================
# CV RESULTS
# ============================================================

oof_score = log_loss(y, oof)

print("\n==========================")
print("RESULTS")
print("==========================")

for i, score in enumerate(fold_scores, 1):
    print(f"Fold {i}: {score:.6f}")

print(
    "\nMean CV:",
    np.mean(fold_scores)
)

print(
    "CV SD:",
    np.std(fold_scores)
)

print(
    "OOF log loss:",
    oof_score
)


# ============================================================
# SUBMISSION
# ============================================================

submission = pd.DataFrame({
    "client_id": test["client_id"],
    "default": test_preds
})

submission.to_csv(
    "catboost_submission.csv",
    index=False
)

print("\nSaved: catboost_submission.csv")

print(submission.head())

print(
    "\nProbability range:",
    submission["default"].min(),
    submission["default"].max()
)