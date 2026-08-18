# ============================================================
# PROVIDER-SIDE SUPERVISED FWA DETECTION
# XGBoost + LightGBM HYBRID
#
# Target:
#   Potential Fraud
#   0 = Legitimate
#   1 = Suspicious
#
# Important:
#   - provider_npi / any detected provider identifier is EXCLUDED
#   - target is EXCLUDED from X
#   - provider_state and provider_type are encoded
#   - NO SMOTE / SMOTENC because the provider dataset is already
#     approximately 50:50
#   - 70/15/15 stratified train/validation/test split
#   - preprocessing is fitted on TRAIN ONLY
#   - validation is used for hybrid-ratio selection
#   - TEST is untouched until final evaluation
#   - one deployable .pkl contains both models + preprocessing
#
# Run from:
#   E:\cts-hack (2)\supervised
#
# Command:
#   python MODELS\trainprovider.py
# ============================================================

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

warnings.filterwarnings("ignore")

# ============================================================
# 1. PATHS / CONFIG
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "DATA"
MODEL_DIR = PROJECT_DIR / "MODELS"

DATA_FILE = DATA_DIR / "Provider_with_potential_fraud.csv"
MODEL_FILE = MODEL_DIR / "provider_fwa_hybrid_model.pkl"

RANDOM_STATE = 42
TARGET_COLUMN = "Potential Fraud"

CLASS_MAPPING = {
    0: "Legitimate",
    1: "Suspicious",
}

# Explicit provider identifiers. These are NEVER allowed into X.
IDENTIFIER_CANDIDATES = {
    "provider_npi",
    "Provider_NPI",
    "PROVIDER_NPI",
    "NPI",
    "Provider_ID",
    "provider_id",
    "ProviderID",
    "provider_identifier",
}

# Known categorical provider features.
KNOWN_CATEGORICAL = {
    "provider_state",
    "provider_type",
}

# ============================================================
# 2. HELPERS
# ============================================================

def print_header(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def evaluate_binary(name, y_true, probability, threshold=0.50):
    prediction = (probability >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true, prediction, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        "name": name,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, prediction)
        ),
        "precision": float(
            precision_score(y_true, prediction, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, prediction, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, prediction, zero_division=0)
        ),
        "macro_precision": float(
            precision_score(
                y_true, prediction, average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true, prediction, average="macro", zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true, prediction, average="macro", zero_division=0
            )
        ),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "log_loss": float(
            log_loss(y_true, probability, labels=[0, 1])
        ),
        "brier_score": float(
            brier_score_loss(y_true, probability)
        ),
        "specificity": float(specificity),
        "npv": float(npv),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def print_metrics(metrics):
    for key in [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "roc_auc",
        "pr_auc",
        "log_loss",
        "brier_score",
        "specificity",
        "false_positive_rate",
        "false_negative_rate",
    ]:
        print(f"  {key:22s}: {metrics[key]:.6f}")


def distribution(name, y):
    counts = pd.Series(y).value_counts().sort_index()
    print(f"\n{name}")
    for cls in [0, 1]:
        count = int(counts.get(cls, 0))
        pct = 100.0 * count / len(y)
        print(
            f"    {cls} - {CLASS_MAPPING[cls]:12s}: "
            f"{count:12,} ({pct:8.3f}%)"
        )


# ============================================================
# 3. START
# ============================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print_header("PROVIDER-SIDE SUPERVISED FWA HYBRID MODEL")

print(f"Project directory : {PROJECT_DIR}")
print(f"Data directory    : {DATA_DIR}")
print(f"Dataset           : {DATA_FILE}")
print(f"Model directory   : {MODEL_DIR}")

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found:\n{DATA_FILE}")

# ============================================================
# 4. LOAD DATA
# ============================================================

print_header("LOADING PROVIDER DATASET")

df = pd.read_csv(DATA_FILE, low_memory=False)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")

print("\nDataset columns:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:3d}. {col}")

# ============================================================
# 5. TARGET VALIDATION
# ============================================================

print_header("TARGET VALIDATION")

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Required target column '{TARGET_COLUMN}' was not found.\n\n"
        "Available columns:\n" +
        "\n".join(f"  - {c}" for c in df.columns)
    )

if df[TARGET_COLUMN].isna().any():
    raise ValueError("Target contains missing values.")

df[TARGET_COLUMN] = pd.to_numeric(
    df[TARGET_COLUMN], errors="raise"
).astype(np.int8)

actual_classes = set(df[TARGET_COLUMN].unique())

if actual_classes != {0, 1}:
    raise ValueError(
        f"Expected target values {{0, 1}}, found {sorted(actual_classes)}"
    )

print(f"Target column : {TARGET_COLUMN}")
print("0 = Legitimate")
print("1 = Suspicious")

print_header("TARGET DISTRIBUTION")
distribution("FULL DATASET", df[TARGET_COLUMN])

# ============================================================
# 6. IDENTIFIER EXCLUSION
# ============================================================
#
# This is a HARD guarantee:
# identifiers are removed before X is constructed.
# provider_npi cannot be used as a feature even if numeric.
# ============================================================

identifier_columns = [
    c for c in df.columns
    if c in IDENTIFIER_CANDIDATES
]

print_header("IDENTIFIER EXCLUSION")

if identifier_columns:
    for c in identifier_columns:
        print(f"EXCLUDED FROM TRAINING: {c}")
else:
    print("No configured provider identifier column was found.")

# Target is also forbidden from features.
forbidden_columns = set(identifier_columns) | {TARGET_COLUMN}

# ============================================================
# 7. FEATURE DETECTION
# ============================================================

categorical_features = []
numerical_features = []
unsupported_columns = []

for c in df.columns:
    if c in forbidden_columns:
        continue

    if (
        c in KNOWN_CATEGORICAL
        or pd.api.types.is_object_dtype(df[c])
        or pd.api.types.is_categorical_dtype(df[c])
    ):
        categorical_features.append(c)
    elif pd.api.types.is_numeric_dtype(df[c]):
        numerical_features.append(c)
    else:
        unsupported_columns.append(c)

categorical_features = list(dict.fromkeys(categorical_features))

if unsupported_columns:
    print("\nUnsupported columns excluded:")
    for c in unsupported_columns:
        print(f"  - {c}")

FEATURE_COLUMNS = numerical_features + categorical_features

if not FEATURE_COLUMNS:
    raise ValueError("No usable provider features were found.")

# HARD LEAKAGE CHECK
leakage_columns = forbidden_columns.intersection(FEATURE_COLUMNS)

if leakage_columns:
    raise RuntimeError(
        "DATA LEAKAGE: forbidden columns entered the feature list: "
        + ", ".join(sorted(leakage_columns))
    )

print_header("FINAL FEATURE CONFIGURATION")

print(f"Model features : {len(FEATURE_COLUMNS)}")
print(f"Numerical      : {len(numerical_features)}")
print(f"Categorical    : {len(categorical_features)}")

print("\nExcluded:")
for c in identifier_columns:
    print(f"  - {c} [IDENTIFIER]")

print(f"  - {TARGET_COLUMN} [TARGET]")

print("\nCategorical features:")
for c in categorical_features:
    print(f"  - {c}")

print("\nNumerical features:")
for c in numerical_features:
    print(f"  - {c}")

# ============================================================
# 8. TRAIN / VALIDATION / TEST
# ============================================================

print_header("TRAIN / VALIDATION / TEST SPLIT")

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df[TARGET_COLUMN],
    random_state=RANDOM_STATE,
)

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df[TARGET_COLUMN],
    random_state=RANDOM_STATE,
)

train_df = train_df.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("Split method: Stratified 70/15/15")
print(f"Training   : {len(train_df):,}")
print(f"Validation : {len(validation_df):,}")
print(f"Test       : {len(test_df):,}")

distribution("TRAIN", train_df[TARGET_COLUMN])
distribution("VALIDATION", validation_df[TARGET_COLUMN])
distribution("TEST", test_df[TARGET_COLUMN])

# ============================================================
# 9. X / y
# ============================================================

X_train = train_df[FEATURE_COLUMNS].copy()
y_train = train_df[TARGET_COLUMN].copy()

X_val = validation_df[FEATURE_COLUMNS].copy()
y_val = validation_df[TARGET_COLUMN].copy()

X_test = test_df[FEATURE_COLUMNS].copy()
y_test = test_df[TARGET_COLUMN].copy()

# ============================================================
# 10. TRAIN-ONLY IMPUTATION
# ============================================================

print_header("TRAIN-ONLY PREPROCESSING")

training_medians = {}

for c in numerical_features:
    train_numeric = pd.to_numeric(X_train[c], errors="coerce")
    median = train_numeric.median()

    if pd.isna(median):
        median = 0.0

    training_medians[c] = float(median)

    X_train[c] = train_numeric.fillna(median)
    X_val[c] = pd.to_numeric(X_val[c], errors="coerce").fillna(median)
    X_test[c] = pd.to_numeric(X_test[c], errors="coerce").fillna(median)

for c in categorical_features:
    X_train[c] = X_train[c].fillna("missing").astype(str)
    X_val[c] = X_val[c].fillna("missing").astype(str)
    X_test[c] = X_test[c].fillna("missing").astype(str)

# ============================================================
# 11. NO SMOTE
# ============================================================
#
# Provider data is already balanced (~50/50).
# Therefore synthetic oversampling is unnecessary.
# All original training rows are retained.
# ============================================================

print_header("CLASS BALANCING")

train_counts = y_train.value_counts().sort_index()
legitimate_count = int(train_counts.get(0, 0))
suspicious_count = int(train_counts.get(1, 0))
ratio = suspicious_count / legitimate_count

print(f"Training legitimate : {legitimate_count:,}")
print(f"Training suspicious  : {suspicious_count:,}")
print(f"Suspicious/legitimate ratio : {ratio:.6f}")

if 0.90 <= ratio <= 1.10:
    print("\nClasses are already balanced.")
    print("SMOTE/SMOTENC: NOT USED")
else:
    print("\nClass imbalance detected.")
    print("This version intentionally does not oversample.")
    print("Class weighting can be enabled if the dataset later becomes imbalanced.")

# ============================================================
# 12. ONE-HOT ENCODING
# ============================================================
#
# Fitted on TRAIN only.
# Unknown validation/test categories are safely ignored.
# ============================================================

try:
    one_hot = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
        dtype=np.float32,
    )
except TypeError:
    one_hot = OneHotEncoder(
        handle_unknown="ignore",
        sparse=True,
        dtype=np.float32,
    )

transformers = []

if categorical_features:
    transformers.append(
        ("categorical", one_hot, categorical_features)
    )

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="passthrough",
)

X_train_final = preprocessor.fit_transform(X_train)
X_val_final = preprocessor.transform(X_val)
X_test_final = preprocessor.transform(X_test)

print(f"\nEncoded train shape : {X_train_final.shape}")
print(f"Encoded validation  : {X_val_final.shape}")
print(f"Encoded test        : {X_test_final.shape}")

# ============================================================
# 13. XGBOOST
# ============================================================

print_header("TRAINING XGBOOST")

xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=8,
    subsample=0.80,
    colsample_bytree=0.80,
    gamma=0.10,
    reg_alpha=0.30,
    reg_lambda=3.0,
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0,
    early_stopping_rounds=40,
)

xgb_model.fit(
    X_train_final,
    y_train,
    eval_set=[(X_val_final, y_val)],
    verbose=False,
)

xgb_best_iteration = getattr(
    xgb_model, "best_iteration", None
)

print("XGBoost training completed.")
if xgb_best_iteration is not None:
    print(f"Best iteration: {xgb_best_iteration}")

# ============================================================
# 14. LIGHTGBM
# ============================================================

print_header("TRAINING LIGHTGBM")

lgb_model = LGBMClassifier(
    objective="binary",
    n_estimators=500,
    learning_rate=0.04,
    num_leaves=31,
    max_depth=7,
    min_child_samples=60,
    subsample=0.80,
    colsample_bytree=0.80,
    reg_alpha=0.30,
    reg_lambda=3.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1,
)

lgb_model.fit(
    X_train_final,
    y_train,
    eval_set=[(X_val_final, y_val)],
    callbacks=[
        early_stopping(
            stopping_rounds=40,
            verbose=False,
        ),
        log_evaluation(0),
    ],
)

lgb_best_iteration = getattr(
    lgb_model, "best_iteration_", None
)

print("LightGBM training completed.")
if lgb_best_iteration is not None:
    print(f"Best iteration: {lgb_best_iteration}")

# ============================================================
# 15. VALIDATION PREDICTIONS
# ============================================================

print_header("INDIVIDUAL MODEL VALIDATION")

xgb_val_prob = xgb_model.predict_proba(X_val_final)[:, 1]
lgb_val_prob = lgb_model.predict_proba(X_val_final)[:, 1]

xgb_val_metrics = evaluate_binary(
    "XGBoost", y_val, xgb_val_prob
)
lgb_val_metrics = evaluate_binary(
    "LightGBM", y_val, lgb_val_prob
)

print("\nXGBoost")
print_metrics(xgb_val_metrics)

print("\nLightGBM")
print_metrics(lgb_val_metrics)

# ============================================================
# 16. HYBRID RATIO SELECTION
# ============================================================
#
# Validation ONLY.
# Test is not used to select the ratio.
# ============================================================

print_header("HYBRID RATIO VALIDATION")

candidate_ratios = [0.30, 0.40, 0.50, 0.60, 0.70]
ratio_results = []

for xgb_ratio in candidate_ratios:
    lgb_ratio = 1.0 - xgb_ratio

    hybrid_prob = (
        xgb_ratio * xgb_val_prob
        + lgb_ratio * lgb_val_prob
    )

    hybrid_pred = (hybrid_prob >= 0.50).astype(np.int8)

    ratio_results.append({
        "xgboost_ratio": xgb_ratio,
        "lightgbm_ratio": lgb_ratio,
        "macro_f1": f1_score(
            y_val, hybrid_pred,
            average="macro",
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_val, hybrid_pred
        ),
        "pr_auc": average_precision_score(
            y_val, hybrid_prob
        ),
        "roc_auc": roc_auc_score(
            y_val, hybrid_prob
        ),
    })

ratio_df = pd.DataFrame(ratio_results)

print(ratio_df.to_string(index=False))

best = (
    ratio_df
    .sort_values(
        ["macro_f1", "pr_auc"],
        ascending=[False, False],
    )
    .iloc[0]
)

XGB_RATIO = float(best["xgboost_ratio"])
LGB_RATIO = float(best["lightgbm_ratio"])

print(f"\nSelected XGBoost ratio : {XGB_RATIO:.2f}")
print(f"Selected LightGBM ratio: {LGB_RATIO:.2f}")

# ============================================================
# 17. FINAL VALIDATION HYBRID
# ============================================================

hybrid_val_prob = (
    XGB_RATIO * xgb_val_prob
    + LGB_RATIO * lgb_val_prob
)

hybrid_val_metrics = evaluate_binary(
    "Hybrid",
    y_val,
    hybrid_val_prob,
)

print_header("FINAL VALIDATION HYBRID")
print_metrics(hybrid_val_metrics)

# ============================================================
# 18. FINAL TEST
# ============================================================
#
# TEST is touched only after model training + ratio selection.
# ============================================================

print_header("FINAL HYBRID TEST EVALUATION")

xgb_test_prob = xgb_model.predict_proba(X_test_final)[:, 1]
lgb_test_prob = lgb_model.predict_proba(X_test_final)[:, 1]

hybrid_test_prob = (
    XGB_RATIO * xgb_test_prob
    + LGB_RATIO * lgb_test_prob
)

hybrid_test_metrics = evaluate_binary(
    "Hybrid",
    y_test,
    hybrid_test_prob,
)

test_pred = (hybrid_test_prob >= 0.50).astype(np.int8)

print_metrics(hybrid_test_metrics)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        test_pred,
        labels=[0, 1],
        target_names=["Legitimate", "Suspicious"],
        zero_division=0,
    )
)

cm = confusion_matrix(
    y_test, test_pred, labels=[0, 1]
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual_Legitimate", "Actual_Suspicious"],
    columns=["Predicted_Legitimate", "Predicted_Suspicious"],
)

print("Confusion matrix:")
print(cm_df.to_string())

# ============================================================
# 19. OVERFITTING CHECK
# ============================================================
#
# This uses ORIGINAL training rows, not synthetic rows.
# No training data is removed from actual model training.
# ============================================================

print_header("OVERFITTING CHECK")

xgb_train_prob = xgb_model.predict_proba(X_train_final)[:, 1]
lgb_train_prob = lgb_model.predict_proba(X_train_final)[:, 1]

hybrid_train_prob = (
    XGB_RATIO * xgb_train_prob
    + LGB_RATIO * lgb_train_prob
)

train_metrics = evaluate_binary(
    "Hybrid Train",
    y_train,
    hybrid_train_prob,
)

train_macro_f1 = train_metrics["macro_f1"]
val_macro_f1 = hybrid_val_metrics["macro_f1"]
test_macro_f1 = hybrid_test_metrics["macro_f1"]

train_val_gap = train_macro_f1 - val_macro_f1
val_test_gap = val_macro_f1 - test_macro_f1

if train_val_gap > 0.10:
    assessment = "WARNING - large train/validation gap"
elif train_val_gap > 0.05:
    assessment = "CAUTION - moderate train/validation gap"
else:
    assessment = "PASS - no large train/validation gap detected"

print(f"Train Macro-F1      : {train_macro_f1:.6f}")
print(f"Validation Macro-F1 : {val_macro_f1:.6f}")
print(f"Test Macro-F1       : {test_macro_f1:.6f}")
print(f"Train-Validation gap: {train_val_gap:.6f}")
print(f"Validation-Test gap : {val_test_gap:.6f}")
print(f"Assessment           : {assessment}")

# ============================================================
# 20. FEATURE IMPORTANCE
# ============================================================

print_header("FEATURE IMPORTANCE")

# ColumnTransformer feature names
try:
    feature_names = preprocessor.get_feature_names_out().tolist()
except Exception:
    feature_names = [
        f"feature_{i}"
        for i in range(X_train_final.shape[1])
    ]

xgb_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": xgb_model.feature_importances_,
}).sort_values(
    "importance",
    ascending=False,
).reset_index(drop=True)

lgb_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": lgb_model.feature_importances_,
}).sort_values(
    "importance",
    ascending=False,
).reset_index(drop=True)

print("\nTop XGBoost features:")
print(xgb_importance.head(20).to_string(index=False))

print("\nTop LightGBM features:")
print(lgb_importance.head(20).to_string(index=False))

# ============================================================
# 21. SAVE ONE DEPLOYABLE PKL
# ============================================================
#
# This is the ONLY model file required for inference.
# It contains:
#   - XGBoost
#   - LightGBM
#   - preprocessing
#   - medians
#   - feature configuration
#   - hybrid ratios
#   - target mapping
#
# NO identifier is part of feature_columns.
# ============================================================

print_header("SAVING ONE HYBRID MODEL")

hybrid_bundle = {
    "model_type": "Provider XGBoost + LightGBM probability hybrid",

    "xgboost_model": xgb_model,
    "lightgbm_model": lgb_model,

    "xgboost_ratio": XGB_RATIO,
    "lightgbm_ratio": LGB_RATIO,

    "preprocessor": preprocessor,
    "training_medians": training_medians,

    "feature_columns": FEATURE_COLUMNS,
    "numerical_features": numerical_features,
    "categorical_features": categorical_features,

    "excluded_identifier_columns": identifier_columns,

    "target_column": TARGET_COLUMN,
    "class_mapping": CLASS_MAPPING,
    "prediction_threshold": 0.50,

    "formula": (
        f"{XGB_RATIO:.2f} * XGBoost_probability + "
        f"{LGB_RATIO:.2f} * LightGBM_probability"
    ),

    "split_method": "Stratified 70/15/15",
    "random_state": RANDOM_STATE,

    "smote_used": False,
    "training_class_balanced": bool(0.90 <= ratio <= 1.10),
}

joblib.dump(
    hybrid_bundle,
    MODEL_FILE,
    compress=3,
)

print(f"Saved deployable model:\n  {MODEL_FILE}")

# ============================================================
# 22. VERIFY SAVED MODEL
# ============================================================

loaded = joblib.load(MODEL_FILE)

if set(loaded["feature_columns"]) & (
    set(identifier_columns) | {TARGET_COLUMN}
):
    raise RuntimeError(
        "SAVED MODEL LEAKAGE CHECK FAILED: "
        "identifier/target found in feature_columns."
    )

print("Saved model verification: PASS")
print("Identifier leakage check: PASS")
print("Target leakage check     : PASS")

# ============================================================
# 23. SAVE EVALUATION ARTIFACTS
# ============================================================

classification_report_df = pd.DataFrame(
    classification_report(
        y_test,
        test_pred,
        labels=[0, 1],
        target_names=["Legitimate", "Suspicious"],
        output_dict=True,
        zero_division=0,
    )
).transpose()

classification_report_df.to_csv(
    MODEL_DIR / "provider_classification_report.csv"
)

cm_df.to_csv(
    MODEL_DIR / "provider_confusion_matrix.csv"
)

ratio_df.to_csv(
    MODEL_DIR / "provider_ensemble_ratio_validation.csv",
    index=False,
)

xgb_importance.to_csv(
    MODEL_DIR / "provider_xgboost_feature_importance.csv",
    index=False,
)

lgb_importance.to_csv(
    MODEL_DIR / "provider_lightgbm_feature_importance.csv",
    index=False,
)

metrics_json = {
    "model": "Provider XGBoost + LightGBM Hybrid",
    "target": TARGET_COLUMN,
    "class_mapping": {
        "0": "Legitimate",
        "1": "Suspicious",
    },
    "dataset_rows": int(len(df)),
    "split": {
        "method": "stratified_70_15_15",
        "train": int(len(train_df)),
        "validation": int(len(validation_df)),
        "test": int(len(test_df)),
    },
    "class_balancing": {
        "smote_used": False,
        "reason": "Provider training data already approximately 50:50",
        "train_legitimate": legitimate_count,
        "train_suspicious": suspicious_count,
        "train_ratio": ratio,
    },
    "ensemble": {
        "xgboost_ratio": XGB_RATIO,
        "lightgbm_ratio": LGB_RATIO,
    },
    "validation": {
        "xgboost": xgb_val_metrics,
        "lightgbm": lgb_val_metrics,
        "hybrid": hybrid_val_metrics,
    },
    "train": train_metrics,
    "test": hybrid_test_metrics,
    "overfitting": {
        "train_macro_f1": train_macro_f1,
        "validation_macro_f1": val_macro_f1,
        "test_macro_f1": test_macro_f1,
        "train_validation_gap": train_val_gap,
        "validation_test_gap": val_test_gap,
        "assessment": assessment,
    },
    "features": {
        "total": len(FEATURE_COLUMNS),
        "numerical": numerical_features,
        "categorical": categorical_features,
        "excluded_identifiers": identifier_columns,
        "target": TARGET_COLUMN,
    },
}

with open(
    MODEL_DIR / "provider_evaluation_metrics.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(metrics_json, f, indent=4)

# ============================================================
# 24. FINAL SUMMARY
# ============================================================

print_header("PROVIDER TRAINING COMPLETED SUCCESSFULLY")

print(f"Total rows       : {len(df):,}")
print(f"Train rows       : {len(train_df):,}")
print(f"Validation rows  : {len(validation_df):,}")
print(f"Test rows        : {len(test_df):,}")

print(f"\nFeatures used    : {len(FEATURE_COLUMNS)}")
print(f"Categorical      : {categorical_features}")
print(f"Identifiers used : NONE")
print(f"Target used as X : NO")
print("SMOTE/SMOTENC     : NOT USED")

print("\nHybrid ratio:")
print(f"  XGBoost  : {XGB_RATIO:.2f}")
print(f"  LightGBM : {LGB_RATIO:.2f}")

print("\nFINAL TEST:")
print_metrics(hybrid_test_metrics)

print("\nOverfitting:")
print(f"  {assessment}")

print("\nONE DEPLOYABLE MODEL:")
print(f"  {MODEL_FILE}")

print("\nTraining completed.")
