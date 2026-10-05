# Quick look at our dataset
import pandas as pd

# Load CSV
df = pd.read_csv('../data/credit_risk_dataset.csv')

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumn Names:\n{list(df.columns)}")
print(f"\nFirst 3 rows:")
print(df.head(3).to_string())
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nBasic Stats (numeric columns):")
print(df.describe().to_string())