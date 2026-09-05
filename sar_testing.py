import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.calibration import calibration_curve
import xgboost as xgb

# 1. Load data
TRAIN_PATH = "train.csv"   # update path as needed
TEST_PATH = "test.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

PAY_COLS = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
BILL_COLS = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
PAYAMT_COLS = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']


# 2. Feature engineering
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Utilisation: bill amount relative to credit limit ---
    for i in range(1, 7):
        df[f'UTIL{i}'] = df[f'BILL_AMT{i}'] / df['LIMIT_BAL'].replace(0, np.nan)
    df['UTIL_AVG'] = df[[f'UTIL{i}' for i in range(1, 7)]].mean(axis=1)
    df['UTIL_LAST'] = df['UTIL1']

    # --- Payment coverage: how much of each bill was paid off ---
    # PAY_AMT_i is the payment made toward the bill that became BILL_AMT_{i+1}
    for i in range(1, 6):
        bill = df[f'BILL_AMT{i+1}']
        pay = df[f'PAY_AMT{i}']
        df[f'PAYRATIO{i}'] = np.where(
            bill > 0, pay / bill,
            np.where(pay > 0, 1.0, np.nan)
        )
    df['PAYRATIO_AVG'] = df[[f'PAYRATIO{i}' for i in range(1, 6)]].mean(axis=1)

    # --- Delinquency severity / frequency / trend ---
    df['PAY_MAX'] = df[PAY_COLS].max(axis=1)
    df['PAY_MEAN'] = df[PAY_COLS].mean(axis=1)
    df['PAY_SUM_LATE'] = df[PAY_COLS].clip(lower=0).sum(axis=1)   # cumulative months-late severity
    df['NUM_MONTHS_LATE'] = (df[PAY_COLS] > 0).sum(axis=1)
    df['WORSENING'] = df['PAY_0'] - df['PAY_6']                    # trend: positive = getting worse

    # --- Bill / payment aggregates ---
    df['BILL_TREND'] = df['BILL_AMT1'] - df['BILL_AMT6']
    df['BILL_AVG'] = df[BILL_COLS].mean(axis=1)
    df['PAYAMT_AVG'] = df[PAYAMT_COLS].mean(axis=1)
    df['PAYAMT_TOTAL'] = df[PAYAMT_COLS].sum(axis=1)

    # --- Clean undocumented categorical codes ---
    df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})  # collapse unknown -> "other"
    df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})                # collapse unknown -> "other"

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


train_fe = engineer(train)
test_fe = engineer(test)

feature_cols = [c for c in train_fe.columns if c not in ['client_id', 'default']]
X = train_fe[feature_cols].fillna(0)
y = train_fe['default']
X_test = test_fe[feature_cols].fillna(0)

print(f"Training rows: {len(X)}, Test rows: {len(X_test)}, Features: {len(feature_cols)}")



# 3. Model definitions
def make_xgb():
    return xgb.XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=2.0,
        eval_metric='logloss',
        random_state=42,
        n_jobs=4,
        tree_method='hist',
    )


def make_logreg():
    return LogisticRegression(max_iter=2000, C=0.5)


# 4. Cross-validated model comparison
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Logistic Regression baseline (needs scaling) ---
ll_lr, auc_lr = [], []
for tr_idx, val_idx in skf.split(X, y):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X.iloc[tr_idx])
    Xval = scaler.transform(X.iloc[val_idx])
    clf = make_logreg()
    clf.fit(Xtr, y.iloc[tr_idx])
    p = clf.predict_proba(Xval)[:, 1]
    ll_lr.append(log_loss(y.iloc[val_idx], p))
    auc_lr.append(roc_auc_score(y.iloc[val_idx], p))
print(f"LogisticRegression  logloss={np.mean(ll_lr):.5f}  AUC={np.mean(auc_lr):.4f}")

# --- XGBoost ---
ll_xgb, auc_xgb = [], []
oof_xgb = np.zeros(len(X))
for tr_idx, val_idx in skf.split(X, y):
    clf = make_xgb()
    clf.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    p = clf.predict_proba(X.iloc[val_idx])[:, 1]
    oof_xgb[val_idx] = p
    ll_xgb.append(log_loss(y.iloc[val_idx], p))
    auc_xgb.append(roc_auc_score(y.iloc[val_idx], p))
print(f"XGBoost             logloss={np.mean(ll_xgb):.5f}  AUC={np.mean(auc_xgb):.4f}")

# --- Calibration check (out-of-fold reliability) ---
frac_pos, mean_pred = calibration_curve(y, oof_xgb, n_bins=10)
print("\nCalibration (predicted vs. observed default rate, OOF):")
for mp, fp in zip(mean_pred, frac_pos):
    print(f"  pred={mp:.3f}  actual={fp:.3f}")


# 5. Feature importance (fit on full training data)
final_model = make_xgb()
final_model.fit(X, y)

importance = (
    pd.Series(final_model.feature_importances_, index=feature_cols)
    .sort_values(ascending=False)
)
print("\nTop 15 feature importances:")
print(importance.head(15))


# 6. Predict on test set and write submission
test_pred = final_model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    'client_id': test_fe['client_id'],
    'default': test_pred,
})
submission.to_csv("submission.csv", index=False)

print(f"\nSaved submission.csv with {len(submission)} rows")
print(f"Predicted probability range: [{test_pred.min():.4f}, {test_pred.max():.4f}], "
      f"mean={test_pred.mean():.4f}")