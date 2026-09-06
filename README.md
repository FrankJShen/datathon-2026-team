# datathon-2026-team (Gamergunk) Stream 1 - Final Submission

## 1. Final Notebook / Methodology Report
**Data Cleaning and Preprocessing**
Missing values in payment ratio columns were intentionally left as `NaN` rather than imputed with `0`. This allows the gradient boosting histogram engine to natively route inactive billing months to the optimal split without falsely signaling a 0% repayment. Sparse categorical variables (`EDUCATION` levels 0, 5, 6, and `MARRIAGE` level 0) were grouped to reduce high-cardinality noise and prevent overfitting.

**Feature Engineering**
*   **Headroom:** Absolute unused credit calculated as `LIMIT_BAL - BILL_AMT1`.
*   **Volatility Tracking:** `BILL_STD` and `PAYAMT_STD` to capture erratic billing and payment behavior across the 6-month window.
*   **Delinquency Trajectory:** Created `MAX_DELAY`, `SUM_DELAYS`, `NUM_MONTHS_LATE`, and `WORSENING` (the difference between `PAY_0` and `PAY_6` to track downward momentum).

**Validation Strategy & Final Model Selection**
*   **Validation:** 10-Fold Stratified Cross-Validation with a locked random seed (`2024`) to ensure reproducible data splits.
*   **Final Model:** A single LightGBM Classifier (gbdt) optimized for binary log-loss.

**Post-Processing & Ensembling**
*   **Pseudo-Labeling:** The base LightGBM model predicted probabilities on the test set. Extreme confidence predictions (>95% and <5%) were appended back to the training set as pseudo-labels.
*   **Calibration:** Isotonic calibration (`CalibratedClassifierCV`) was applied out-of-fold to align probability distributions.
*   **Base Rate Alignment:** A SciPy `brentq` optimizer was used to shift the final predictions in log-odds space, ensuring the overall predicted mean matches the historical default base rate of 22.12%.
*   **Clipping:** Predictions were hard-clipped between 0.015 and 0.985 to prevent infinite log-loss penalties.

## 2. Reproduction Instructions
1. Install the exact environment dependencies by running `pip install -r requirements.txt`.
2. Ensure `train.csv` and `test.csv` are placed in the same root directory as the script.
3. Run `new_x_master_pipeline.py`. 
4. The script automatically executes feature generation, pseudo-labeling, 10-fold CV training, and generates `augmented_train.csv` (the intermediate dataset) alongside the final submission CSV.
5. **Note:** The LightGBM model is locked to `n_jobs=-1`. Hardware thread counts may cause microscopic floating-point variations deep in the decimal range depending on the execution machine. (Different CPUs may result in slightly different output)

## 3. Final Model Information
*   **Algorithm:** Single LightGBM Classifier (Gradient Boosting Decision Tree).   
*   **Hyperparameters:** `learning_rate`: 0.036019, `num_leaves`: 51, `max_depth`: 5, `min_child_weight`: 6.709387, `subsample`: 0.664436, `colsample_bytree`: 0.957112, `reg_alpha`: 0.053511, `reg_lambda`: 0.314483, `n_estimators`: 700.

## 4. Required Disclosure
*   **External Datasets:** None used.
*   **Pretrained Models:** None used.
*   **AI Tools:** Google Gemini was utilised strictly as a coding assistant to refine the SciPy log-odds base rate alignment function and to troubleshoot pandas categorical data type mappings.
*   **Manual Modifications:** No manual modification of the final predictions occurred outside of the automated mathematical log-odds base rate alignment.