
import pandas as pd

# 1. Load dataset
df = pd.read_csv("star_classification.csv")

print("Original dataset shape:", df.shape)

# 2. Feature engineering - colour indices
df["u_g"] = df["u"] - df["g"]
df["g_r"] = df["g"] - df["r"]
df["r_i"] = df["r"] - df["i"]
df["i_z"] = df["i"] - df["z"]

# 3. Select input features
features = [
    "alpha",
    "delta",
    "u",
    "g",
    "r",
    "i",
    "z",
    "u_g",
    "g_r",
    "r_i",
    "i_z",
    "redshift"
]

# 4. Select target
X = df[features]
y = df["class"]

# 5. Display information
print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())

print("\nMissing values:")
print(X.isnull().sum().sum())

# 6. Create processed dataset for inspection
processed_df = X.copy()
processed_df["class"] = y

processed_df.to_csv("processed_stellar_data.csv", index=False)

print("\nProcessed dataset saved successfully.")
