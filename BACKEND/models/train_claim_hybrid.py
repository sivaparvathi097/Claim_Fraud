# ============================================================
# CLAIM-SIDE FWA SUPERVISED HYBRID MODEL
# ============================================================
#
# Models:
#   1. XGBoost
#   2. LightGBM
#   3. Probability-level Hybrid Ensemble
#
# Directory:
#
# Supervised/
# ├── DATA/
# │   └── claimsfinal_labeled.csv
# │
# └── MODELS/
#     └── trainclaim.py
#
# Run from Supervised:
#
#     python MODELS\trainclaim.py
#
#
# TARGET:
#   0 = Legitimate
#   1 = Fraud
#   2 = Waste
#   3 = Abuse
#
#
# IMPORTANT:
#
# - Claim_ID is NOT used as a feature.
# - Provider_ID is NOT used as a feature.
# - BENE_ID is NOT used as a feature.
# - Target is NOT used as a feature.
# - Claim_Year is NOT used as a feature.
# - Claim_Year is used ONLY for temporal splitting.
#
# SPLIT:
#
#   TRAIN      : 2015 - 2021
#   VALIDATION : 2022
#   TEST       : 2023
#
# SMOTENC:
#   Applied ONLY to training data.
#
# TEST:
#   Never SMOTENC'ed.
#   Never used for ensemble-ratio selection.
#
# ENSEMBLE:
#
#   Final probability =
#       XGBoost probability * XGB ratio
#       +
#       LightGBM probability * LGB ratio
#
# Ratio selected using VALIDATION Macro-F1.
#
# ============================================================


from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    top_k_accuracy_score,
)

from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTENC

from xgboost import XGBClassifier

from lightgbm import (
    LGBMClassifier,
    early_stopping,
    log_evaluation,
)


warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT DIRECTORIES
# ============================================================
#
# trainclaim.py is inside:
#
#     Supervised/MODELS/
#
# Therefore:
#
#     parent       = Supervised/MODELS
#     parent.parent = Supervised
#
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_DIR = SCRIPT_DIR.parent

DATA_DIR = PROJECT_DIR / "DATA"

MODEL_DIR = PROJECT_DIR / "MODELS"

DATA_FILE = DATA_DIR / "claimsfinal_labeled.csv"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("\n" + "=" * 90)
print("CLAIM-SIDE FWA SUPERVISED HYBRID TRAINING")
print("=" * 90)

print(f"\nProject directory : {PROJECT_DIR}")
print(f"Data directory    : {DATA_DIR}")
print(f"Dataset           : {DATA_FILE}")
print(f"Model directory   : {MODEL_DIR}")


# ============================================================
# 2. CONFIGURATION
# ============================================================

TARGET_COLUMN =  "Target_Label"

YEAR_COLUMN = "Claim_Year"

RANDOM_STATE = 42


# ------------------------------------------------------------
# Temporal split
# ------------------------------------------------------------

TRAIN_START_YEAR = 2015
TRAIN_END_YEAR = 2021

VALIDATION_YEAR = 2022

TEST_YEAR = 2023


# ------------------------------------------------------------
# FWA class mapping
# ------------------------------------------------------------

CLASS_NAMES = [
    "Legitimate",
    "Fraud",
    "Waste",
    "Abuse",
]

CLASS_IDS = [0, 1, 2, 3]


# ============================================================
# 3. VERIFY DATASET EXISTS
# ============================================================

if not DATA_FILE.exists():

    raise FileNotFoundError(
        "\nDataset not found.\n"
        f"Expected:\n{DATA_FILE}\n\n"
        "Required structure:\n"
        "Supervised/\n"
        "├── DATA/\n"
        "│   └── claimsfinal_labeled.csv\n"
        "└── MODELS/\n"
        "    └── trainclaim.py\n"
    )


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\n" + "=" * 90)
print("LOADING DATASET")
print("=" * 90)

df = pd.read_csv(DATA_FILE)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# 5. REQUIRED COLUMNS
# ============================================================

required_columns = {
    "Provider_ID",
    "BENE_ID",
    "Claim_ID",
    "Claim_Type",
    "Claim_Year",
    TARGET_COLUMN,
}


missing_columns = (
    required_columns
    - set(df.columns)
)


if missing_columns:

    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(
            f"  - {column}"
            for column in sorted(missing_columns)
        )
    )


# ============================================================
# 6. TARGET VALIDATION
# ============================================================

if df[TARGET_COLUMN].isna().any():

    raise ValueError(
        "Target column contains missing values."
    )


# Make target integer
df[TARGET_COLUMN] = (
    pd.to_numeric(
        df[TARGET_COLUMN],
        errors="raise"
    )
    .astype(int)
)


actual_classes = set(
    df[TARGET_COLUMN].unique()
)


expected_classes = {
    0,
    1,
    2,
    3,
}


if actual_classes != expected_classes:

    raise ValueError(
        f"\nIncorrect target classes.\n"
        f"Expected: {expected_classes}\n"
        f"Found   : {actual_classes}"
    )


# ============================================================
# 7. CLAIM YEAR VALIDATION
# ============================================================

df[YEAR_COLUMN] = pd.to_numeric(
    df[YEAR_COLUMN],
    errors="raise"
).astype(int)


print("\nClaim year distribution:")

print(
    df[YEAR_COLUMN]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# 8. IDENTIFIER VALIDATION
# ============================================================

print("\n" + "=" * 90)
print("IDENTIFIER CHECK")
print("=" * 90)

print(
    "Duplicate Claim_IDs : "
    f"{df['Claim_ID'].duplicated().sum():,}"
)

print(
    "Duplicate full rows : "
    f"{df.duplicated().sum():,}"
)


# ============================================================
# 9. ORIGINAL TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 90)
print("OVERALL TARGET DISTRIBUTION")
print("=" * 90)

overall_counts = (
    df[TARGET_COLUMN]
    .value_counts()
    .sort_index()
)

overall_distribution = pd.DataFrame({
    "Class": CLASS_NAMES,
    "Count": [
        overall_counts.get(i, 0)
        for i in CLASS_IDS
    ],
})

overall_distribution["Percentage"] = (
    overall_distribution["Count"]
    / len(df)
    * 100
)

overall_distribution["Percentage"] = (
    overall_distribution["Percentage"]
    .round(3)
)

print(
    overall_distribution.to_string(
        index=False
    )
)


# ============================================================
# 10. IDENTIFIERS THAT MUST NEVER ENTER THE MODEL
# ============================================================

IDENTIFIER_COLUMNS = [
    "Claim_ID",
    "Provider_ID",
    "BENE_ID",
]


# ============================================================
# 11. COLUMNS THAT MUST NOT ENTER TRAINING
# ============================================================
#
# Claim_Year is deliberately excluded from features.
#
# It is used ONLY to create the temporal split.
#
# This prevents the model from learning time directly.
# ============================================================

EXCLUDED_COLUMNS = (
    IDENTIFIER_COLUMNS
    + [
        TARGET_COLUMN,
        YEAR_COLUMN,
    ]
)


FEATURE_COLUMNS = [
    column
    for column in df.columns
    if column not in EXCLUDED_COLUMNS
]


# ============================================================
# 12. VERIFY NO FORBIDDEN COLUMNS
# ============================================================

for forbidden in EXCLUDED_COLUMNS:

    if forbidden in FEATURE_COLUMNS:

        raise RuntimeError(
            f"DATA LEAKAGE ERROR: "
            f"{forbidden} is present in feature list."
        )


print("\n" + "=" * 90)
print("FEATURE SELECTION")
print("=" * 90)

print(
    f"Total original columns : {len(df.columns)}"
)

print(
    f"Excluded columns       : {len(EXCLUDED_COLUMNS)}"
)

print(
    f"Model features         : {len(FEATURE_COLUMNS)}"
)

print("\nExcluded from model:")

for column in EXCLUDED_COLUMNS:

    print(
        f"  - {column}"
    )

print("\nFeatures entering model:")

for column in FEATURE_COLUMNS:

    print(
        f"  - {column}"
    )


# ============================================================
# 13. IDENTIFY CATEGORICAL FEATURES
# ============================================================
#
# Claim_Type is categorical.
#
# Other features are expected to be numerical in this dataset.
#
# If another object/category column appears, the script stops
# rather than silently treating it as numeric.
# ============================================================

categorical_features = [
    "Claim_Type"
]


unexpected_object_columns = [
    column
    for column in FEATURE_COLUMNS
    if (
        df[column].dtype == "object"
        and column not in categorical_features
    )
]


if unexpected_object_columns:

    raise ValueError(
        "\nUnexpected categorical/object columns found:\n"
        + "\n".join(
            f"  - {column}"
            for column in unexpected_object_columns
        )
        + "\n\nAdd these columns to categorical_features "
        "after verifying their meaning."
    )


numerical_features = [
    column
    for column in FEATURE_COLUMNS
    if column not in categorical_features
]


print("\nCategorical features:")

for column in categorical_features:

    print(
        f"  - {column}"
    )


print(
    f"\nNumerical feature count: "
    f"{len(numerical_features)}"
)


# ============================================================
# 14. TEMPORAL SPLIT
# ============================================================
#
# IMPORTANT:
#
# 2014 has only one record.
#
# We exclude it.
#
# Train:
#     2015-2021
#
# Validation:
#     2022
#
# Test:
#     2023
#
# There is NO overlap.
# ============================================================

train_df = df[
    (
        df[YEAR_COLUMN]
        >= TRAIN_START_YEAR
    )
    &
    (
        df[YEAR_COLUMN]
        <= TRAIN_END_YEAR
    )
].copy()


validation_df = df[
    df[YEAR_COLUMN]
    == VALIDATION_YEAR
].copy()


test_df = df[
    df[YEAR_COLUMN]
    == TEST_YEAR
].copy()


# ============================================================
# 15. REMOVE UNEXPECTED YEARS
# ============================================================

used_rows = (
    len(train_df)
    + len(validation_df)
    + len(test_df)
)


unused_rows = len(df) - used_rows


print("\n" + "=" * 90)
print("TEMPORAL SPLIT")
print("=" * 90)

print(
    f"Training rows      : {len(train_df):,}"
)

print(
    f"Validation rows    : {len(validation_df):,}"
)

print(
    f"Test rows          : {len(test_df):,}"
)

print(
    f"Excluded rows      : {unused_rows:,}"
)


# ============================================================
# 16. VERIFY EXACT YEAR BOUNDARIES
# ============================================================

print("\nTraining years:")

print(
    sorted(
        train_df[YEAR_COLUMN]
        .unique()
        .tolist()
    )
)


print("\nValidation years:")

print(
    sorted(
        validation_df[YEAR_COLUMN]
        .unique()
        .tolist()
    )
)


print("\nTest years:")

print(
    sorted(
        test_df[YEAR_COLUMN]
        .unique()
        .tolist()
    )
)


# ============================================================
# 17. VERIFY NO ROW OVERLAP
# ============================================================

train_claim_ids = set(
    train_df["Claim_ID"]
)

validation_claim_ids = set(
    validation_df["Claim_ID"]
)

test_claim_ids = set(
    test_df["Claim_ID"]
)


if (
    train_claim_ids
    & validation_claim_ids
):

    raise RuntimeError(
        "Claim_ID overlap detected between "
        "training and validation."
    )


if (
    train_claim_ids
    & test_claim_ids
):

    raise RuntimeError(
        "Claim_ID overlap detected between "
        "training and test."
    )


if (
    validation_claim_ids
    & test_claim_ids
):

    raise RuntimeError(
        "Claim_ID overlap detected between "
        "validation and test."
    )


print(
    "\nNo Claim_ID overlap between "
    "train / validation / test."
)


# ============================================================
# 18. TARGET DISTRIBUTION FUNCTION
# ============================================================

def show_target_distribution(
    name,
    data
):

    counts = (
        data[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    result = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Count": [
            counts.get(i, 0)
            for i in CLASS_IDS
        ],
    })

    result["Percentage"] = (
        result["Count"]
        / len(data)
        * 100
    )

    result["Percentage"] = (
        result["Percentage"]
        .round(3)
    )

    print(
        f"\n{name}"
    )

    print(
        result.to_string(
            index=False
        )
    )


show_target_distribution(
    "TRAINING DISTRIBUTION",
    train_df
)

show_target_distribution(
    "VALIDATION DISTRIBUTION",
    validation_df
)

show_target_distribution(
    "TEST DISTRIBUTION",
    test_df
)


# ============================================================
# 19. MAKE X / y
# ============================================================

X_train = (
    train_df[FEATURE_COLUMNS]
    .copy()
)

y_train = (
    train_df[TARGET_COLUMN]
    .copy()
)


X_validation = (
    validation_df[FEATURE_COLUMNS]
    .copy()
)

y_validation = (
    validation_df[TARGET_COLUMN]
    .copy()
)


X_test = (
    test_df[FEATURE_COLUMNS]
    .copy()
)

y_test = (
    test_df[TARGET_COLUMN]
    .copy()
)


# ============================================================
# 20. HANDLE MISSING VALUES
# ============================================================
#
# Numerical:
#     median calculated ONLY from training.
#
# Categorical:
#     "missing"
# ============================================================

for column in numerical_features:

    median_value = (
        X_train[column]
        .median()
    )

    X_train[column] = (
        X_train[column]
        .fillna(median_value)
    )

    X_validation[column] = (
        X_validation[column]
        .fillna(median_value)
    )

    X_test[column] = (
        X_test[column]
        .fillna(median_value)
    )


for column in categorical_features:

    X_train[column] = (
        X_train[column]
        .fillna("missing")
        .astype(str)
    )

    X_validation[column] = (
        X_validation[column]
        .fillna("missing")
        .astype(str)
    )

    X_test[column] = (
        X_test[column]
        .fillna("missing")
        .astype(str)
    )


# ============================================================
# 21. ENCODE CATEGORICAL DATA FOR SMOTENC
# ============================================================

ordinal_encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)


X_train_encoded = X_train.copy()

X_validation_encoded = (
    X_validation.copy()
)

X_test_encoded = X_test.copy()


X_train_encoded[
    categorical_features
] = ordinal_encoder.fit_transform(
    X_train[
        categorical_features
    ]
)


X_validation_encoded[
    categorical_features
] = ordinal_encoder.transform(
    X_validation[
        categorical_features
    ]
)


X_test_encoded[
    categorical_features
] = ordinal_encoder.transform(
    X_test[
        categorical_features
    ]
)


# ============================================================
# 22. CONTROLLED SMOTENC
# ============================================================
#
# SMOTENC ONLY TRAINING.
#
# We do NOT balance the classes to 25/25/25/25.
#
# Target training counts:
#
#   Fraud -> 60,000
#   Waste -> 80,000
#   Abuse -> 40,000
#
# Legitimate is NOT oversampled.
#
# If an original class already exceeds its requested target,
# its count is left unchanged.
# ============================================================

print("\n" + "=" * 90)
print("CONTROLLED SMOTENC")
print("=" * 90)


categorical_indices = [
    X_train_encoded.columns.get_loc(
        column
    )
    for column in categorical_features
]


sampling_strategy = {
    1: 60_000,
    2: 80_000,
    3: 40_000,
}


# Never downsample.
for class_id in list(
    sampling_strategy.keys()
):

    current_count = int(
        (
            y_train
            == class_id
        ).sum()
    )

    if current_count >= sampling_strategy[class_id]:

        sampling_strategy[class_id] = (
            current_count
        )


print(
    "\nSMOTENC target counts:"
)

for class_id, count in (
    sampling_strategy.items()
):

    print(
        f"  {CLASS_NAMES[class_id]:12s}: "
        f"{count:,}"
    )


smotenc = SMOTENC(
    categorical_features=categorical_indices,
    sampling_strategy=sampling_strategy,
    random_state=RANDOM_STATE,
    k_neighbors=5,
)


X_train_smote, y_train_smote = (
    smotenc.fit_resample(
        X_train_encoded,
        y_train
    )
)


print(
    "\nAfter SMOTENC:"
)

smote_counts = (
    pd.Series(
        y_train_smote
    )
    .value_counts()
    .sort_index()
)


for class_id in CLASS_IDS:

    print(
        f"  {CLASS_NAMES[class_id]:12s}: "
        f"{smote_counts.get(class_id, 0):,}"
    )


# ============================================================
# 23. ONE-HOT ENCODE CLAIM_TYPE
# ============================================================

one_hot_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "claim_type",
            one_hot_encoder,
            categorical_features
        )
    ],
    remainder="passthrough"
)


X_train_final = (
    preprocessor.fit_transform(
        X_train_smote
    )
)


X_validation_final = (
    preprocessor.transform(
        X_validation_encoded
    )
)


X_test_final = (
    preprocessor.transform(
        X_test_encoded
    )
)


# ============================================================
# 24. FEATURE NAMES
# ============================================================

encoded_claim_type_names = (
    preprocessor
    .named_transformers_[
        "claim_type"
    ]
    .get_feature_names_out(
        categorical_features
    )
    .tolist()
)


feature_names = (
    encoded_claim_type_names
    + numerical_features
)


print(
    "\nFinal model feature count: "
    f"{len(feature_names)}"
)


# ============================================================
# 25. CONVERT TO DATAFRAME
# ============================================================

X_train_final = pd.DataFrame(
    X_train_final,
    columns=feature_names
)

X_validation_final = pd.DataFrame(
    X_validation_final,
    columns=feature_names
)

X_test_final = pd.DataFrame(
    X_test_final,
    columns=feature_names
)


# ============================================================
# 26. CLASS WEIGHTS
# ============================================================
#
# Since SMOTENC has already increased minority classes,
# use moderated class weights rather than extremely aggressive
# "balanced" weighting.
#
# sqrt(balanced weight)
# ============================================================

original_class_weights = (
    compute_class_weight(
        class_weight="balanced",
        classes=np.array(
            CLASS_IDS
        ),
        y=y_train
    )
)


moderated_class_weights = {
    int(class_id): float(
        np.sqrt(weight)
    )
    for class_id, weight in zip(
        CLASS_IDS,
        original_class_weights
    )
}


print("\n" + "=" * 90)
print("MODERATED CLASS WEIGHTS")
print("=" * 90)


for class_id in CLASS_IDS:

    print(
        f"{CLASS_NAMES[class_id]:12s}: "
        f"{moderated_class_weights[class_id]:.4f}"
    )


sample_weights = np.array([
    moderated_class_weights[
        int(label)
    ]
    for label in y_train_smote
])


# ============================================================
# 27. TRAIN XGBOOST
# ============================================================

print("\n" + "=" * 90)
print("TRAINING XGBOOST")
print("=" * 90)


xgb_model = XGBClassifier(

    objective="multi:softprob",

    num_class=4,

    n_estimators=700,

    learning_rate=0.05,

    max_depth=8,

    min_child_weight=5,

    subsample=0.85,

    colsample_bytree=0.85,

    gamma=0.1,

    reg_alpha=0.2,

    reg_lambda=2.0,

    tree_method="hist",

    eval_metric="mlogloss",

    random_state=RANDOM_STATE,

    n_jobs=-1,

    verbosity=1,
)


xgb_model.fit(

    X_train_final,

    y_train_smote,

    sample_weight=sample_weights,

    eval_set=[
        (
            X_validation_final,
            y_validation
        )
    ],

    verbose=False,
)


print(
    "XGBoost training completed."
)


# ============================================================
# 28. TRAIN LIGHTGBM
# ============================================================

print("\n" + "=" * 90)
print("TRAINING LIGHTGBM")
print("=" * 90)


lgb_model = LGBMClassifier(

    objective="multiclass",

    num_class=4,

    n_estimators=1000,

    learning_rate=0.03,

    num_leaves=63,

    max_depth=-1,

    min_child_samples=50,

    subsample=0.85,

    colsample_bytree=0.85,

    reg_alpha=0.2,

    reg_lambda=2.0,

    random_state=RANDOM_STATE,

    n_jobs=-1,

    verbosity=-1,
)


lgb_model.fit(

    X_train_final,

    y_train_smote,

    sample_weight=sample_weights,

    eval_set=[
        (
            X_validation_final,
            y_validation
        )
    ],

    callbacks=[
        early_stopping(
            stopping_rounds=60,
            verbose=False
        ),
        log_evaluation(0)
    ],
)


print(
    "LightGBM training completed."
)


# ============================================================
# 29. VALIDATION PREDICTIONS
# ============================================================

print("\n" + "=" * 90)
print("SELECTING XGBOOST + LIGHTGBM ENSEMBLE RATIO")
print("=" * 90)


xgb_validation_proba = (
    xgb_model.predict_proba(
        X_validation_final
    )
)


lgb_validation_proba = (
    lgb_model.predict_proba(
        X_validation_final
    )
)


# ============================================================
# 30. SEARCH ENSEMBLE RATIOS
# ============================================================
#
# XGBoost ratio:
#
# 0.30
# 0.40
# 0.50
# 0.60
# 0.70
#
# LightGBM ratio:
#
# 1 - XGBoost ratio
#
# Best ratio = highest VALIDATION Macro-F1.
#
# TEST DATA IS NOT USED.
# ============================================================

candidate_ratios = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
]


ratio_results = []


for xgb_ratio in candidate_ratios:

    lgb_ratio = (
        1.0
        - xgb_ratio
    )


    validation_proba = (
        xgb_ratio
        * xgb_validation_proba
        +
        lgb_ratio
        * lgb_validation_proba
    )


    validation_pred = (
        np.argmax(
            validation_proba,
            axis=1
        )
    )


    macro_f1 = f1_score(
        y_validation,
        validation_pred,
        average="macro",
        zero_division=0
    )


    weighted_f1 = f1_score(
        y_validation,
        validation_pred,
        average="weighted",
        zero_division=0
    )


    balanced_accuracy = (
        balanced_accuracy_score(
            y_validation,
            validation_pred
        )
    )


    ratio_results.append({

        "xgboost_ratio":
            xgb_ratio,

        "lightgbm_ratio":
            lgb_ratio,

        "macro_f1":
            macro_f1,

        "weighted_f1":
            weighted_f1,

        "balanced_accuracy":
            balanced_accuracy,
    })


ratio_results_df = pd.DataFrame(
    ratio_results
)


print(
    "\nValidation ratio comparison:"
)

print(
    ratio_results_df.to_string(
        index=False
    )
)


# ============================================================
# 31. SELECT BEST RATIO
# ============================================================

best_ratio = (
    ratio_results_df
    .sort_values(
        by="macro_f1",
        ascending=False
    )
    .iloc[0]
)


XGB_RATIO = float(
    best_ratio[
        "xgboost_ratio"
    ]
)


LIGHTGBM_RATIO = float(
    best_ratio[
        "lightgbm_ratio"
    ]
)


print("\nSelected ratio:")

print(
    f"XGBoost  : "
    f"{XGB_RATIO:.2f}"
)

print(
    f"LightGBM : "
    f"{LIGHTGBM_RATIO:.2f}"
)


# ============================================================
# 32. TEST PREDICTIONS
# ============================================================
#
# IMPORTANT:
#
# Test data has never been SMOTENC'ed.
#
# Test distribution remains original.
#
# The test set is used ONLY ONCE here for final evaluation.
# ============================================================

print("\n" + "=" * 90)
print("FINAL TEST EVALUATION")
print("=" * 90)


xgb_test_proba = (
    xgb_model.predict_proba(
        X_test_final
    )
)


lgb_test_proba = (
    lgb_model.predict_proba(
        X_test_final
    )
)


ensemble_test_proba = (
    XGB_RATIO
    * xgb_test_proba
    +
    LIGHTGBM_RATIO
    * lgb_test_proba
)


ensemble_test_pred = (
    np.argmax(
        ensemble_test_proba,
        axis=1
    )
)


# ============================================================
# 33. BASIC METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    ensemble_test_pred
)


balanced_accuracy = (
    balanced_accuracy_score(
        y_test,
        ensemble_test_pred
    )
)


macro_precision = (
    precision_score(
        y_test,
        ensemble_test_pred,
        average="macro",
        zero_division=0
    )
)


macro_recall = (
    recall_score(
        y_test,
        ensemble_test_pred,
        average="macro",
        zero_division=0
    )
)


macro_f1 = (
    f1_score(
        y_test,
        ensemble_test_pred,
        average="macro",
        zero_division=0
    )
)


weighted_precision = (
    precision_score(
        y_test,
        ensemble_test_pred,
        average="weighted",
        zero_division=0
    )
)


weighted_recall = (
    recall_score(
        y_test,
        ensemble_test_pred,
        average="weighted",
        zero_division=0
    )
)


weighted_f1 = (
    f1_score(
        y_test,
        ensemble_test_pred,
        average="weighted",
        zero_division=0
    )
)


test_log_loss = log_loss(
    y_test,
    ensemble_test_proba,
    labels=CLASS_IDS
)


# ============================================================
# 34. ROC-AUC
# ============================================================

try:

    macro_roc_auc = roc_auc_score(
        y_test,
        ensemble_test_proba,
        multi_class="ovr",
        average="macro"
    )


    weighted_roc_auc = roc_auc_score(
        y_test,
        ensemble_test_proba,
        multi_class="ovr",
        average="weighted"
    )

except ValueError:

    macro_roc_auc = None

    weighted_roc_auc = None


# ============================================================
# 35. MULTICLASS PR-AUC
# ============================================================

y_test_one_hot = np.eye(4)[
    y_test.to_numpy()
]


per_class_pr_auc = {}


for class_id in CLASS_IDS:

    score = (
        average_precision_score(
            y_test_one_hot[
                :, class_id
            ],
            ensemble_test_proba[
                :, class_id
            ]
        )
    )


    per_class_pr_auc[
        CLASS_NAMES[class_id]
    ] = score


macro_pr_auc = np.mean(
    list(
        per_class_pr_auc.values()
    )
)


# ============================================================
# 36. TOP-K METRICS
# ============================================================

top_2_accuracy = (
    top_k_accuracy_score(
        y_test,
        ensemble_test_proba,
        k=2,
        labels=CLASS_IDS
    )
)


top_3_accuracy = (
    top_k_accuracy_score(
        y_test,
        ensemble_test_proba,
        k=3,
        labels=CLASS_IDS
    )
)


# ============================================================
# 37. CLASSIFICATION REPORT
# ============================================================

classification_report_dict = (
    classification_report(
        y_test,
        ensemble_test_pred,
        labels=CLASS_IDS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
)


classification_report_text = (
    classification_report(
        y_test,
        ensemble_test_pred,
        labels=CLASS_IDS,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ============================================================
# 38. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    ensemble_test_pred,
    labels=CLASS_IDS
)


cm_df = pd.DataFrame(
    cm,
    index=[
        f"Actual_{name}"
        for name in CLASS_NAMES
    ],
    columns=[
        f"Predicted_{name}"
        for name in CLASS_NAMES
    ]
)


# ============================================================
# 39. PRINT MAIN METRICS
# ============================================================

print("\n" + "=" * 90)
print("FINAL HYBRID MODEL METRICS")
print("=" * 90)


print(
    f"\nAccuracy             : "
    f"{accuracy:.6f}"
)


print(
    f"Balanced Accuracy    : "
    f"{balanced_accuracy:.6f}"
)


print(
    f"\nMacro Precision      : "
    f"{macro_precision:.6f}"
)


print(
    f"Macro Recall         : "
    f"{macro_recall:.6f}"
)


print(
    f"Macro F1             : "
    f"{macro_f1:.6f}"
)


print(
    f"\nWeighted Precision   : "
    f"{weighted_precision:.6f}"
)


print(
    f"Weighted Recall      : "
    f"{weighted_recall:.6f}"
)


print(
    f"Weighted F1          : "
    f"{weighted_f1:.6f}"
)


print(
    f"\nMacro ROC-AUC        : "
    f"{macro_roc_auc}"
)


print(
    f"Weighted ROC-AUC     : "
    f"{weighted_roc_auc}"
)


print(
    f"\nMacro PR-AUC         : "
    f"{macro_pr_auc:.6f}"
)


print(
    f"Top-2 Accuracy       : "
    f"{top_2_accuracy:.6f}"
)


print(
    f"Top-3 Accuracy       : "
    f"{top_3_accuracy:.6f}"
)


print(
    f"\nLog Loss             : "
    f"{test_log_loss:.6f}"
)


# ============================================================
# 40. PER-CLASS PR-AUC
# ============================================================

print("\n" + "=" * 90)
print("PER-CLASS PR-AUC")
print("=" * 90)


for class_name, score in (
    per_class_pr_auc.items()
):

    print(
        f"{class_name:12s}: "
        f"{score:.6f}"
    )


# ============================================================
# 41. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 90)
print("CLASSIFICATION REPORT")
print("=" * 90)

print(
    classification_report_text
)


# ============================================================
# 42. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 90)
print("CONFUSION MATRIX")
print("=" * 90)

print(
    cm_df.to_string()
)


# ============================================================
# 43. PER-CLASS PERFORMANCE
# ============================================================

print("\n" + "=" * 90)
print("PER-CLASS PERFORMANCE")
print("=" * 90)


for class_id, class_name in (
    enumerate(CLASS_NAMES)
):

    precision = (
        classification_report_dict[
            class_name
        ]["precision"]
    )


    recall = (
        classification_report_dict[
            class_name
        ]["recall"]
    )


    f1 = (
        classification_report_dict[
            class_name
        ]["f1-score"]
    )


    support = (
        classification_report_dict[
            class_name
        ]["support"]
    )


    print(
        f"\n{class_name}"
    )


    print(
        f"  Precision : "
        f"{precision:.6f}"
    )


    print(
        f"  Recall    : "
        f"{recall:.6f}"
    )


    print(
        f"  F1        : "
        f"{f1:.6f}"
    )


    print(
        f"  Support   : "
        f"{int(support):,}"
    )


# ============================================================
# 44. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 90)
print("FEATURE IMPORTANCE")
print("=" * 90)


xgb_importance = pd.DataFrame({

    "feature":
        feature_names,

    "importance":
        xgb_model.feature_importances_

})


xgb_importance = (
    xgb_importance
    .sort_values(
        by="importance",
        ascending=False
    )
    .reset_index(drop=True)
)


lgb_importance = pd.DataFrame({

    "feature":
        feature_names,

    "importance":
        lgb_model.feature_importances_

})


lgb_importance = (
    lgb_importance
    .sort_values(
        by="importance",
        ascending=False
    )
    .reset_index(drop=True)
)


print("\nTop 20 XGBoost features:")

print(
    xgb_importance
    .head(20)
    .to_string(
        index=False
    )
)


print("\nTop 20 LightGBM features:")

print(
    lgb_importance
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# 45. SAVE FEATURE IMPORTANCE
# ============================================================

xgb_importance.to_csv(
    MODEL_DIR
    / "xgboost_feature_importance.csv",
    index=False
)


lgb_importance.to_csv(
    MODEL_DIR
    / "lightgbm_feature_importance.csv",
    index=False
)


# ============================================================
# 46. SAVE CONFUSION MATRIX
# ============================================================

cm_df.to_csv(
    MODEL_DIR
    / "confusion_matrix.csv"
)


# ============================================================
# 47. SAVE CLASSIFICATION REPORT
# ============================================================

classification_report_df = (
    pd.DataFrame(
        classification_report_dict
    ).transpose()
)


classification_report_df.to_csv(
    MODEL_DIR
    / "classification_report.csv"
)


# ============================================================
# 48. SAVE ENSEMBLE RATIO RESULTS
# ============================================================

ratio_results_df.to_csv(
    MODEL_DIR
    / "ensemble_ratio_validation.csv",
    index=False
)


# ============================================================
# 49. SAVE METRICS
# ============================================================

metrics = {

    "model":
        "XGBoost + LightGBM Hybrid",

    "xgboost_ratio":
        XGB_RATIO,

    "lightgbm_ratio":
        LIGHTGBM_RATIO,

    "split": {

        "train":
            "2015-2021",

        "validation":
            "2022",

        "test":
            "2023"

    },

    "rows": {

        "original":
            int(len(df)),

        "train":
            int(len(train_df)),

        "train_after_smotenc":
            int(len(X_train_smote)),

        "validation":
            int(len(validation_df)),

        "test":
            int(len(test_df)),

        "excluded":
            int(unused_rows)

    },

    "accuracy":
        float(accuracy),

    "balanced_accuracy":
        float(balanced_accuracy),

    "macro_precision":
        float(macro_precision),

    "macro_recall":
        float(macro_recall),

    "macro_f1":
        float(macro_f1),

    "weighted_precision":
        float(weighted_precision),

    "weighted_recall":
        float(weighted_recall),

    "weighted_f1":
        float(weighted_f1),

    "macro_roc_auc":
        (
            None
            if macro_roc_auc is None
            else float(
                macro_roc_auc
            )
        ),

    "weighted_roc_auc":
        (
            None
            if weighted_roc_auc is None
            else float(
                weighted_roc_auc
            )
        ),

    "macro_pr_auc":
        float(
            macro_pr_auc
        ),

    "per_class_pr_auc":
        {
            key: float(value)
            for key, value
            in per_class_pr_auc.items()
        },

    "top_2_accuracy":
        float(
            top_2_accuracy
        ),

    "top_3_accuracy":
        float(
            top_3_accuracy
        ),

    "log_loss":
        float(
            test_log_loss
        ),

    "excluded_identifiers":
        IDENTIFIER_COLUMNS,

    "excluded_from_training":
        EXCLUDED_COLUMNS,

    "target_column":
        TARGET_COLUMN,

    "classes":
        {
            "0":
                "Legitimate",

            "1":
                "Fraud",

            "2":
                "Waste",

            "3":
                "Abuse"
        },

    "smotenc_strategy":
        {
            str(key):
                int(value)
            for key, value
            in sampling_strategy.items()
        },

    "moderated_class_weights":
        {
            str(key):
                float(value)
            for key, value
            in moderated_class_weights.items()
        }

}


with open(
    MODEL_DIR
    / "evaluation_metrics.json",
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ============================================================
# 50. SAVE ONE DEPLOYABLE HYBRID MODEL
# ============================================================
#
# IMPORTANT:
#
# XGBoost and LightGBM are two INTERNAL components of the
# hybrid model. They are NOT saved as separate .pkl files.
#
# Everything required for inference is packaged into ONE file:
#
#     claim_fwa_hybrid_model.pkl
#
# The bundle contains:
#
#   - XGBoost model
#   - LightGBM model
#   - OrdinalEncoder
#   - OneHot/ColumnTransformer preprocessor
#   - training medians for numerical columns
#   - feature list
#   - categorical/numerical feature lists
#   - ensemble ratio
#   - class mapping
#
# SMOTENC is NOT saved because it is a TRAINING-ONLY operation.
# It must never be applied to incoming claims during inference.
#
# ============================================================

print("\n" + "=" * 90)
print("PACKAGING ONE DEPLOYABLE HYBRID MODEL")
print("=" * 90)


# ------------------------------------------------------------
# Store the training medians used during preprocessing.
# These are required when a future claim contains a missing
# numerical value.
# ------------------------------------------------------------

training_medians = {
    column: float(
        X_train[column].median()
    )
    for column in numerical_features
}


# ------------------------------------------------------------
# Create ONE complete model bundle.
#
# The two tree models remain internal components of the
# probability-level hybrid ensemble.
# ------------------------------------------------------------

hybrid_model_bundle = {

    "model_type":
        "XGBoost + LightGBM probability-level hybrid",

    "xgboost_model":
        xgb_model,

    "lightgbm_model":
        lgb_model,

    "xgboost_ratio":
        XGB_RATIO,

    "lightgbm_ratio":
        LIGHTGBM_RATIO,

    "ordinal_encoder":
        ordinal_encoder,

    "preprocessor":
        preprocessor,

    "training_medians":
        training_medians,

    "feature_columns":
        FEATURE_COLUMNS,

    "categorical_features":
        categorical_features,

    "numerical_features":
        numerical_features,

    "target_column":
        TARGET_COLUMN,

    "year_column":
        YEAR_COLUMN,

    "class_mapping":
        {
            0: "Legitimate",
            1: "Fraud",
            2: "Waste",
            3: "Abuse",
        },

    "class_ids":
        CLASS_IDS,

    "class_names":
        CLASS_NAMES,

    "excluded_columns":
        EXCLUDED_COLUMNS,

    "excluded_identifiers":
        IDENTIFIER_COLUMNS,

    "prediction_rule":
        "argmax(weighted_probability)",

    "formula":
        (
            "final_probability = "
            f"{XGB_RATIO:.2f} * XGBoost_probability + "
            f"{LIGHTGBM_RATIO:.2f} * LightGBM_probability"
        ),

    "training_split":
        {
            "train":
                "2015-2021",

            "validation":
                "2022",

            "test":
                "2023",
        },

    "smotenc_training_only":
        True,

}


# ------------------------------------------------------------
# Remove old component .pkl files created by the previous
# version of this training script.
#
# This prevents the MODELS folder from appearing to contain
# multiple deployable models.
# ------------------------------------------------------------

old_component_files = [
    "claim_xgboost_model.pkl",
    "claim_lightgbm_model.pkl",
    "claim_ordinal_encoder.pkl",
    "claim_preprocessor.pkl",
    "claim_smotenc.pkl",
]


for old_file in old_component_files:

    old_path = MODEL_DIR / old_file

    if old_path.exists():

        old_path.unlink()

        print(
            f"Removed old component file: "
            f"{old_file}"
        )


# ------------------------------------------------------------
# Save ONE .pkl file.
# ------------------------------------------------------------

HYBRID_MODEL_FILE = (
    MODEL_DIR
    / "claim_fwa_hybrid_model.pkl"
)


joblib.dump(
    hybrid_model_bundle,
    HYBRID_MODEL_FILE,
    compress=3
)


print(
    "\nSingle deployable model saved:"
)

print(
    f"  {HYBRID_MODEL_FILE}"
)


# ------------------------------------------------------------
# Verify the saved file can be loaded immediately.
# ------------------------------------------------------------

loaded_hybrid_model = joblib.load(
    HYBRID_MODEL_FILE
)


required_bundle_keys = {
    "xgboost_model",
    "lightgbm_model",
    "xgboost_ratio",
    "lightgbm_ratio",
    "ordinal_encoder",
    "preprocessor",
    "training_medians",
    "feature_columns",
    "categorical_features",
    "numerical_features",
    "class_mapping",
}


missing_bundle_keys = (
    required_bundle_keys
    - set(
        loaded_hybrid_model.keys()
    )
)


if missing_bundle_keys:

    raise RuntimeError(
        "Saved hybrid model is incomplete. "
        f"Missing keys: {missing_bundle_keys}"
    )


print(
    "  ✓ Saved hybrid model verified successfully."
)


# ============================================================
# 51. SAVE ENSEMBLE CONFIGURATION
# ============================================================

ensemble_config = {

    "model_type":
        "XGBoost + LightGBM probability ensemble",

    "xgboost_ratio":
        XGB_RATIO,

    "lightgbm_ratio":
        LIGHTGBM_RATIO,

    "formula":
        (
            "final_probability = "
            f"{XGB_RATIO:.2f} * "
            "xgboost_probability + "
            f"{LIGHTGBM_RATIO:.2f} * "
            "lightgbm_probability"
        ),

    "prediction_rule":
        "argmax(final_probability)",

    "classes":
        {
            "0":
                "Legitimate",

            "1":
                "Fraud",

            "2":
                "Waste",

            "3":
                "Abuse"
        }

}


with open(
    MODEL_DIR
    / "ensemble_config.json",
    "w"
) as file:

    json.dump(
        ensemble_config,
        file,
        indent=4
    )


# ============================================================
# 52. SAVE FEATURE CONFIGURATION
# ============================================================

feature_config = {

    "training_features":
        FEATURE_COLUMNS,

    "categorical_features":
        categorical_features,

    "numerical_features":
        numerical_features,

    "excluded_identifiers":
        IDENTIFIER_COLUMNS,

    "excluded_columns":
        EXCLUDED_COLUMNS,

    "target_column":
        TARGET_COLUMN,

    "year_column_used_for_split":
        YEAR_COLUMN

}


with open(
    MODEL_DIR
    / "training_features.json",
    "w"
) as file:

    json.dump(
        feature_config,
        file,
        indent=4
    )


# ============================================================
# 53. SAVE SPLIT INFORMATION
# ============================================================

split_information = {

    "train":
        {
            "start_year":
                TRAIN_START_YEAR,

            "end_year":
                TRAIN_END_YEAR,

            "rows":
                len(train_df)
        },

    "validation":
        {
            "year":
                VALIDATION_YEAR,

            "rows":
                len(validation_df)
        },

    "test":
        {
            "year":
                TEST_YEAR,

            "rows":
                len(test_df)
        },

    "excluded_years":
        sorted(
            set(
                df[YEAR_COLUMN]
            )
            -
            set(
                train_df[YEAR_COLUMN]
            )
            -
            set(
                validation_df[YEAR_COLUMN]
            )
            -
            set(
                test_df[YEAR_COLUMN]
            )
        )

}


with open(
    MODEL_DIR
    / "split_information.json",
    "w"
) as file:

    json.dump(
        split_information,
        file,
        indent=4
    )


# ============================================================
# 54. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 90)


print("\nDeployable model:")
print("  claim_fwa_hybrid_model.pkl")
print("  (contains XGBoost + LightGBM + preprocessing + ensemble configuration)")


print("\nEnsemble ratio:")
print(
    f"  XGBoost  : {XGB_RATIO:.2f}"
)

print(
    f"  LightGBM : {LIGHTGBM_RATIO:.2f}"
)


print("\nData handling:")
print(
    "  ✓ Claim_ID excluded"
)

print(
    "  ✓ Provider_ID excluded"
)

print(
    "  ✓ BENE_ID excluded"
)

print(
    "  ✓ Target excluded"
)

print(
    "  ✓ Claim_Year excluded from model features"
)

print(
    "  ✓ Temporal train/validation/test split"
)

print(
    "  ✓ SMOTENC only on training data"
)

print(
    "  ✓ Validation kept untouched"
)

print(
    "  ✓ Test kept completely untouched"
)

print(
    "  ✓ Controlled minority oversampling"
)

print(
    "  ✓ Moderated class weighting"
)

print(
    "  ✓ Ensemble ratio selected on validation Macro-F1"
)

print(
    "  ✓ Final metrics calculated on original test distribution"
)


print("\nFinal Test Metrics:")

print(
    f"  Accuracy          : "
    f"{accuracy:.6f}"
)

print(
    f"  Balanced Accuracy : "
    f"{balanced_accuracy:.6f}"
)

print(
    f"  Macro Precision   : "
    f"{macro_precision:.6f}"
)

print(
    f"  Macro Recall      : "
    f"{macro_recall:.6f}"
)

print(
    f"  Macro F1          : "
    f"{macro_f1:.6f}"
)

print(
    f"  Weighted F1       : "
    f"{weighted_f1:.6f}"
)

print(
    f"  Macro ROC-AUC     : "
    f"{macro_roc_auc}"
)

print(
    f"  Macro PR-AUC      : "
    f"{macro_pr_auc:.6f}"
)


print("\nModel and evaluation files saved to:")

print(
    f"  {MODEL_DIR}"
)


print("\nDeployable artifact:")
print("  ✓ claim_fwa_hybrid_model.pkl  ← ONE model file")
print("\n" + "=" * 90)
print("DONE")
print("=" * 90)
