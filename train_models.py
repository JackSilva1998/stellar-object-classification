
import pandas as pd
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# --------------------------------------------------
# 1. Load training and testing data
# --------------------------------------------------

train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

X_train = train_df.drop(columns=["class"])
y_train = train_df["class"]

X_test = test_df.drop(columns=["class"])
y_test = test_df["class"]

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# --------------------------------------------------
# 2. Define five classification algorithms
# --------------------------------------------------

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GaussianNB())
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ))
    ])
}

# --------------------------------------------------
# 3. Train models and calculate evaluation metrics
# --------------------------------------------------

os.makedirs("saved_models", exist_ok=True)

results = []

for name, model in models.items():

    print(f"\nTraining {name}...")

    # Train model
    model.fit(X_train, y_train)

    # Predict test data
    y_pred = model.predict(X_test)

    # Calculate probabilities for multiclass AUC
    y_prob = model.predict_proba(X_test)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    auc = roc_auc_score(
        y_test,
        y_prob,
        multi_class="ovr",
        average="macro",
        labels=model.classes_
    )

    precision = precision_score(
        y_test, y_pred, average="macro"
    )

    recall = recall_score(
        y_test, y_pred, average="macro"
    )

    f1 = f1_score(
        y_test, y_pred, average="macro"
    )

    mcc = matthews_corrcoef(y_test, y_pred)

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    })

    # Save trained model
    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, os.path.join("saved_models", filename))

    print(f"{name} training completed.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC:      {auc:.4f}")
    print(f"Precision:{precision:.4f}")
    print(f"Recall:   {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MCC:      {mcc:.4f}")

# --------------------------------------------------
# 4. Save model comparison table
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv("model_comparison.csv", index=False)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

print("\nAll trained models saved in saved_models/")
print("Comparison saved as model_comparison.csv")
