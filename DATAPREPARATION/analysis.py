import pandas as pd

df = pd.read_csv("../DATA-KAGGLE/Train_Beneficiarydata-1542865627584.csv")

df["DOD"] = pd.to_datetime(df["DOD"], errors="coerce")

latest_dod = df["DOD"].max()

print("Latest DOD:", latest_dod)
print("Latest DOD year:", latest_dod.year)