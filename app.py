
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Stellar Classification",
    page_icon="🌌",
    layout="wide"
)

st.title("🌌 Stellar Object Classification")
st.write(
    "Machine Learning application to classify astronomical "
    "objects into Galaxy, Quasar (QSO), and Star."
)

st.divider()

# --------------------------------------------------
# 2. Model selection
# --------------------------------------------------

model_options = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib"
}

selected_model = st.selectbox(
    "Select Classification Model",
    list(model_options.keys())
)

# --------------------------------------------------
# 3. Load saved model
# --------------------------------------------------

model_path = os.path.join(
    "saved_models",
    model_options[selected_model]
)

if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

model = joblib.load(model_path)

st.success(f"{selected_model} model loaded successfully.")

# --------------------------------------------------
# 4. Upload test dataset
# --------------------------------------------------

st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV file containing features and class",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("Uploaded Dataset Preview")
    st.dataframe(df.head())

    # --------------------------------------------------
    # 5. Validate uploaded dataset
    # --------------------------------------------------

    if "class" not in df.columns:
        st.error("The uploaded CSV must contain a 'class' column.")
        st.stop()

    X = df.drop(columns=["class"])
    y_actual = df["class"]

    required_features = list(model.feature_names_in_)

    missing_features = [
        feature for feature in required_features
        if feature not in X.columns
    ]

    if missing_features:
        st.error(
            f"Missing required features: {missing_features}"
        )
        st.stop()

    # Keep the exact feature order used during training
    X = X[required_features]

    if y_actual.isnull().any():
        st.error("The target column contains missing values.")
        st.stop()

    if not set(y_actual.unique()).issubset(set(model.classes_)):
        st.error("The target column contains unknown class labels.")
        st.stop()

    # --------------------------------------------------
    # 6. Generate predictions
    # --------------------------------------------------

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    # --------------------------------------------------
    # 7. Calculate evaluation metrics
    # --------------------------------------------------

    accuracy = accuracy_score(y_actual, y_pred)

    auc = roc_auc_score(
        y_actual,
        y_prob,
        multi_class="ovr",
        average="macro",
        labels=model.classes_
    )

    precision = precision_score(
        y_actual, y_pred, average="macro"
    )

    recall = recall_score(
        y_actual, y_pred, average="macro"
    )

    f1 = f1_score(
        y_actual, y_pred, average="macro"
    )

    mcc = matthews_corrcoef(y_actual, y_pred)

    # --------------------------------------------------
    # 8. Display metrics
    # --------------------------------------------------

    st.divider()
    st.subheader("Model Evaluation Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("AUC", f"{auc:.4f}")
    col3.metric("Precision", f"{precision:.4f}")

    col4, col5, col6 = st.columns(3)

    col4.metric("Recall", f"{recall:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC", f"{mcc:.4f}")

    # --------------------------------------------------
    # 9. Confusion matrix
    # --------------------------------------------------

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_actual,
        y_pred,
        labels=model.classes_
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_,
        ax=ax
    )

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title(f"{selected_model} - Confusion Matrix")

    st.pyplot(fig)
    plt.close(fig)

    # --------------------------------------------------
    # 10. Classification report
    # --------------------------------------------------

    st.subheader("Classification Report")

    report = classification_report(
        y_actual,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df.round(4))

    # --------------------------------------------------
    # 11. Prediction results
    # --------------------------------------------------

    st.subheader("Prediction Results")

    prediction_df = X.copy()
    prediction_df["Actual Class"] = y_actual
    prediction_df["Predicted Class"] = y_pred

    st.dataframe(prediction_df.head(100))

    csv = prediction_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Prediction Results",
        data=csv,
        file_name="stellar_predictions.csv",
        mime="text/csv"
    )
