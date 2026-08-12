import pandas as pd

# ============================================================
# 1. INPUT FILE
# ============================================================
input_file = "Train_Inpatientdata-1542865627584.csv"

# ============================================================
# 2. OUTPUT FILE
# ============================================================
output_file = "Train_Inpatientdata_final.csv"

# ============================================================
# 3. READ DATA
# ============================================================
df = pd.read_csv(input_file, dtype=str, low_memory=False)

print("Original shape:", df.shape)

# ============================================================
# 4. CALCULATE NUMBER OF DAYS STAYED
# ============================================================

df["ClaimStartDt"] = pd.to_datetime(
    df["ClaimStartDt"],
    errors="coerce"
)

df["ClaimEndDt"] = pd.to_datetime(
    df["ClaimEndDt"],
    errors="coerce"
)

df["No.of.days.stayed"] = (
    df["ClaimEndDt"] - df["ClaimStartDt"]
).dt.days

# ============================================================
# 5. COLUMNS TO DELETE
# ============================================================

columns_to_delete = [
    "AttendingPhysician",
    "OperatingPhysician",
    "OtherPhysician",

    "AdmissionDt",
    "DischargeDt",

    "ClaimStartDt",
    "ClaimEndDt",

    "ClmDiagnosisCode_1",
    "ClmDiagnosisCode_2",
    "ClmDiagnosisCode_3",
    "ClmDiagnosisCode_4",
    "ClmDiagnosisCode_5",
    "ClmDiagnosisCode_6",
    "ClmDiagnosisCode_7",
    "ClmDiagnosisCode_8",
    "ClmDiagnosisCode_9",
    "ClmDiagnosisCode_10",

    "ClmProcedureCode_1",
    "ClmProcedureCode_2",
    "ClmProcedureCode_3",
    "ClmProcedureCode_4",
    "ClmProcedureCode_5",
    "ClmProcedureCode_6"
]

# ============================================================
# 6. DELETE THE COLUMNS
# ============================================================

df.drop(
    columns=columns_to_delete,
    errors="ignore",
    inplace=True
)

# ============================================================
# 7. SAVE FINAL DATASET
# ============================================================

df.to_csv(output_file, index=False)

# ============================================================
# 8. CHECK RESULT
# ============================================================

print("\n✅ Processing completed successfully!")
print("Final shape:", df.shape)
print("\nFinal columns:")
print(df.columns.tolist())

print("\nFirst 5 values of No.of.days.stayed:")
print(df["No.of.days.stayed"].head())