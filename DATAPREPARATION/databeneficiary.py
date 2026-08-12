import pandas as pd
import os


# ============================================================
# 1. FILE PATHS
# ============================================================

# Location of this Python file:
# FRAUD DETECTION/DATAPREPARATION/databeneficiary.py

base_folder = os.path.dirname(os.path.abspath(__file__))


# Original beneficiary CSV
input_file = os.path.join(
    base_folder,
    "..",
    "DATA-KAGGLE",
    "Train_Beneficiarydata-1542865627584.csv"
)


# Final cleaned CSV
output_file = os.path.join(
    base_folder,
    "CleanedBeneficiary.csv"
)


# ============================================================
# 2. LOAD THE CSV DATASET
# ============================================================

print("=" * 60)
print("LOADING BENEFICIARY DATASET")
print("=" * 60)

df = pd.read_csv(input_file)

print("Dataset loaded successfully!")

print("Original number of rows    :", df.shape[0])
print("Original number of columns :", df.shape[1])


# ============================================================
# 3. CONVERT DOB TO DATE FORMAT
# ============================================================

df["DOB"] = pd.to_datetime(
    df["DOB"],
    errors="coerce"
)

print("\nDOB converted to date format.")


# ============================================================
# 4. CALCULATE AGE AS OF 31-DEC-2009
# ============================================================

# We calculate the age of every beneficiary
# as of December 31, 2009.
#
# DOD is NOT used for calculating age.

reference_date = pd.Timestamp("2009-12-31")

df["Age"] = (
    reference_date.year
    - df["DOB"].dt.year
    - (
        (reference_date.month < df["DOB"].dt.month)
        |
        (
            (reference_date.month == df["DOB"].dt.month)
            &
            (reference_date.day < df["DOB"].dt.day)
        )
    )
)

print("Age calculated successfully.")
print("Age reference date: 31-Dec-2009")


# ============================================================
# 5. DISPLAY AGE SAMPLE
# ============================================================

print("\n" + "=" * 60)
print("AGE SAMPLE")
print("=" * 60)

print(
    df[["DOB", "Age"]].head(10)
)


# ============================================================
# 6. CHECK AGE STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("AGE STATISTICS")
print("=" * 60)

print(
    df["Age"].describe()
)


# ============================================================
# 7. CHECK GENDER
# ============================================================

# IMPORTANT:
#
# Gender is NOT converted into Male/Female.
#
# Original numerical values are retained.
#
# 1 = Male
# 2 = Female

print("\n" + "=" * 60)
print("GENDER VALUES")
print("=" * 60)

print(
    df["Gender"].value_counts(dropna=False)
)


# ============================================================
# 8. CHRONIC CONDITION COLUMNS
# ============================================================

chronic_columns = [

    "ChronicCond_Alzheimer",

    "ChronicCond_Heartfailure",

    "ChronicCond_KidneyDisease",

    "ChronicCond_Cancer",

    "ChronicCond_ObstrPulmonary",

    "ChronicCond_Depression",

    "ChronicCond_Diabetes",

    "ChronicCond_IschemicHeart",

    "ChronicCond_Osteoporasis",

    "ChronicCond_rheumatoidarthritis",

    "ChronicCond_stroke"
]


# ============================================================
# 9. FINANCIAL COLUMNS
# ============================================================

financial_columns = [

    "IPAnnualReimbursementAmt",

    "IPAnnualDeductibleAmt",

    "OPAnnualReimbursementAmt",

    "OPAnnualDeductibleAmt"
]


# ============================================================
# 10. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [

    # Beneficiary information
    "BeneID",
    "Age",
    "Gender",

    # Chronic conditions
    "ChronicCond_Alzheimer",
    "ChronicCond_Heartfailure",
    "ChronicCond_KidneyDisease",
    "ChronicCond_Cancer",
    "ChronicCond_ObstrPulmonary",
    "ChronicCond_Depression",
    "ChronicCond_Diabetes",
    "ChronicCond_IschemicHeart",
    "ChronicCond_Osteoporasis",
    "ChronicCond_rheumatoidarthritis",
    "ChronicCond_stroke",

    # Financial features
    "IPAnnualReimbursementAmt",
    "IPAnnualDeductibleAmt",
    "OPAnnualReimbursementAmt",
    "OPAnnualDeductibleAmt"
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    print("\n" + "=" * 60)
    print("ERROR: REQUIRED COLUMNS ARE MISSING")
    print("=" * 60)

    for column in missing_columns:
        print("-", column)

    raise ValueError(
        "Some required columns are missing from the dataset."
    )


# ============================================================
# 11. SELECT ONLY REQUIRED COLUMNS
# ============================================================

df_final = df[
    required_columns
].copy()


# ============================================================
# 12. CONVERT FINANCIAL COLUMNS TO NUMERIC
# ============================================================

for column in financial_columns:

    df_final[column] = pd.to_numeric(
        df_final[column],
        errors="coerce"
    )


# ============================================================
# 13. CHECK FINAL COLUMNS
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATASET")
print("=" * 60)

print("Final number of rows    :", df_final.shape[0])
print("Final number of columns :", df_final.shape[1])


print("\nFinal columns:")

for column in df_final.columns:
    print(column)


# ============================================================
# 14. VERIFY DOB AND DOD ARE REMOVED
# ============================================================

print("\n" + "=" * 60)
print("REMOVED COLUMN CHECK")
print("=" * 60)

if "DOB" not in df_final.columns:
    print("DOB: Removed successfully")
else:
    print("DOB: ERROR - still present")


if "DOD" not in df_final.columns:
    print("DOD: Removed successfully")
else:
    print("DOD: ERROR - still present")


# ============================================================
# 15. CHECK GENDER VALUES
# ============================================================

print("\n" + "=" * 60)
print("FINAL GENDER VALUES")
print("=" * 60)

print(
    df_final["Gender"].value_counts(dropna=False)
)


# ============================================================
# 16. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing_values = df_final.isnull().sum()

print(missing_values)


# ============================================================
# 17. CHECK DUPLICATE BENEFICIARIES
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE BeneID CHECK")
print("=" * 60)

duplicate_count = df_final[
    "BeneID"
].duplicated().sum()

print(
    "Duplicate BeneID values:",
    duplicate_count
)


# ============================================================
# 18. DISPLAY FIRST 5 ROWS
# ============================================================

print("\n" + "=" * 60)
print("FIRST 5 ROWS OF FINAL DATASET")
print("=" * 60)

print(
    df_final.head()
)


# ============================================================
# 19. SAVE CLEANED DATASET
# ============================================================

print("\n" + "=" * 60)
print("SAVING CLEANED DATASET")
print("=" * 60)

try:

    df_final.to_csv(
        output_file,
        index=False
    )

    print("File saved successfully!")

except PermissionError:

    print("\nERROR: Permission denied.")
    print(
        "Please close CleanedBeneficiary.csv if it is open "
        "in Excel or another program."
    )

    raise


# ============================================================
# 20. FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print(
    "File name: CleanedBeneficiary.csv"
)

print(
    "Saved location:",
    output_file
)

print(
    "Final dataset shape:",
    df_final.shape
)

print("\nThe final dataset contains:")

print("- BeneID")
print("- Age (calculated as of 31-Dec-2009)")
print("- Gender (original 1/2 values)")
print("- 11 chronic-condition features")
print("- 4 reimbursement/deductible features")

print("\nRemoved from final dataset:")

print("- DOB")
print("- DOD")