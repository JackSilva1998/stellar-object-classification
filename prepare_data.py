
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load processed dataset
df = pd.read_csv("processed_stellar_data.csv")

print("Dataset shape:", df.shape)

# 2. Separate features and target
X = df.drop(columns=["class"])
y = df["class"]

# 3. Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nClass encoding:")
for index, class_name in enumerate(label_encoder.classes_):
    print(class_name, "=", index)

# 4. Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

# 5. Display dataset sizes
print("\nTraining dataset:", X_train.shape)
print("Testing dataset:", X_test.shape)

# 6. Display class distribution
print("\nTraining class distribution:")
print(pd.Series(label_encoder.inverse_transform(y_train)).value_counts())

print("\nTesting class distribution:")
print(pd.Series(label_encoder.inverse_transform(y_test)).value_counts())

# 7. Save test dataset with original class labels
test_data = X_test.copy()
test_data["class"] = label_encoder.inverse_transform(y_test)

test_data.to_csv("test_data.csv", index=False)

# 8. Save training dataset for reproducibility
train_data = X_train.copy()
train_data["class"] = label_encoder.inverse_transform(y_train)

train_data.to_csv("train_data.csv", index=False)

print("\nTraining and testing CSV files saved successfully.")
