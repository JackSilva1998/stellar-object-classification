
import pandas as pd
from pathlib import Path

file_path = Path("star_classification.csv")
df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nData types and non-null counts:")
df.info()

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nClass counts:")
print(df["class"].value_counts())

print("\nClass percentages:")
print((df["class"].value_counts(normalize=True) * 100).round(2))

print("\nUnique target labels:")
print(df["class"].unique())
