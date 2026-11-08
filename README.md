# Stellar Object Classification Using Machine Learning

## 1. Project Overview

This project focuses on classifying astronomical objects into three categories:

- **GALAXY** – Galaxies
- **QSO** – Quasars
- **STAR** – Stars

The classification is performed using five supervised machine learning algorithms. Their performance is evaluated using six classification metrics.

A Streamlit web application is developed to allow users to upload a test dataset, select a trained classification model, and view its evaluation results.

## 2. Problem Statement

Astronomical surveys collect large volumes of data about celestial objects. Identifying whether an observed object is a galaxy, quasar, or star is an important classification task.

The objective of this project is to develop and compare machine learning classification models using astronomical measurements from the Sloan Digital Sky Survey (SDSS).

## 3. Dataset Description

**Dataset:** Stellar Classification Dataset – SDSS17

**Source:** Kaggle  
https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

**Original survey:** Sloan Digital Sky Survey (SDSS)  
https://www.sdss.org/dr17/

Dataset characteristics:

- Total instances: 100,000. It was divided into 80% training data (80,000 instances) and 20% testing data (20,000 instances) using stratified sampling.
- Total original columns: 18
- Target variable: `class`
- Target classes: GALAXY, QSO, STAR
- Missing values: None
- Duplicate records: None

### Class distribution

| Class | Instances | Percentage |
|---|---:|---:|
| GALAXY | 59,445 | 59.44% |
| STAR | 21,594 | 21.59% |
| QSO | 18,961 | 18.96% |
| **Total** | **100,000** | **100%** |

## 4. Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the original SDSS17 dataset.
2. Checked for missing values and duplicate records.
3. Excluded identifier and survey metadata columns from the model inputs.
4. Selected astronomical measurements such as right ascension, declination, and photometric magnitudes.
5. Created colour-index features:
   - `u_g = u - g`
   - `g_r = g - r`
   - `r_i = r - i`
   - `i_z = i - z`
6. Included `redshift` as an input feature.
7. Prepared the final dataset with 12 input features and one target column.
8. The dataset was split into training and testing sets using an 80:20 ratio:
   - Training dataset (train_data.csv): 80,000 records (80%).
   - Testing dataset (test_data.csv): 20,000 records (20%).
9. Applied feature scaling through machine learning pipelines for the applicable algorithms.

## 5. Machine Learning Algorithms

Five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

All five models were trained using the same training dataset and evaluated using the same testing dataset.

## 6. Evaluation Metrics

The following metrics were used:

- **Accuracy:** Measures the proportion of correctly classified instances.
- **AUC:** Measures the ability of the classifier to distinguish between classes. Multiclass AUC was calculated using One-vs-Rest with macro averaging.
- **Precision:** Measures the reliability of positive predictions.
- **Recall:** Measures how effectively actual instances are identified.
- **F1 Score:** Harmonic mean of precision and recall.
- **MCC:** Matthews Correlation Coefficient, which measures the overall quality of classification predictions.

Macro averaging was used for precision, recall, F1 score, and AUC.

## 7. Model Performance Comparison

The models achieved the following results on the test dataset:

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.95630 | 0.987932 | 0.953611 | 0.947138 | 0.949656 | 0.922583 |
| Decision Tree | 0.96815 | 0.971715 | 0.963284 | 0.963112 | 0.963198 | 0.943527 |
| KNN | 0.94775 | 0.978929 | 0.945846 | 0.938108 | 0.941759 | 0.907066 |
| Naive Bayes | 0.72550 | 0.944412 | 0.786614 | 0.620628 | 0.544350 | 0.509917 |
| SVM | 0.96335 | 0.989547 | 0.961171 | 0.956991 | 0.958369 | 0.935249 |

## 8. Observations and Results

### Logistic Regression
Achieved an accuracy of 95.63% and an AUC of 98.79%. It provided strong classification performance across the three classes.

### Decision Tree
Achieved the highest accuracy of 96.82%, highest macro precision, highest macro recall, highest F1 score of 96.32%, and highest MCC of 0.9435. It demonstrated the best overall performance based on these metrics.

### K-Nearest Neighbors
Achieved an accuracy of 94.78%. It provided good classification performance, although it performed below Decision Tree, SVM, and Logistic Regression.

### Gaussian Naive Bayes
Achieved an accuracy of 72.55%. Its comparatively lower performance may be related to the conditional independence assumption of Naive Bayes and correlations among the input features.

### Support Vector Machine
Achieved an accuracy of 96.34% and the highest AUC of 98.95%. It demonstrated strong class discrimination and competitive overall classification performance.

### Overall Conclusion

The Decision Tree classifier achieved the best overall performance based on accuracy, macro precision, macro recall, F1 score, and MCC. SVM achieved the highest AUC.

Therefore, Decision Tree is selected as the best overall classifier for this experiment, while SVM is also a strong alternative.

## 9. Streamlit Application

A Streamlit application was developed to demonstrate the trained models.

Application features:

- Upload a CSV test dataset.
- Select one of the five trained classification models.
- Load the corresponding saved model.
- Generate predictions on uploaded data.
- Display accuracy, AUC, precision, recall, F1 score, and MCC.
- Display the confusion matrix.
- Display the classification report.
- Download prediction results.

The application loads previously trained models and does not retrain them using uploaded test data.

## 10. Project Structure

```text
ML_Assignment_2/
│
├── app.py
├── train_models.py
├── preprocess.py
├── prepare_data.py
├── inspect_dataset.py
├── requirements.txt
├── README.md
│
├── test_data.csv
├── model_comparison.csv
│
└── saved_models/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

## 11. Project Links

- **GitHub Repository:** [[Stellar Object Classification](https://github.com/JackSilva1998/stellar-object-classification)]
- **Live Streamlit Application:** [[Stellar Classification App](https://stellar-object-classification-asvagwt4fbtoxnvortjk73.streamlit.app/)]
- **Dataset:** https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

## 12. Limitations

The `redshift` feature is highly informative for astronomical classification and can make the task easier. Since redshift is a spectroscopic measurement, the results should not be interpreted as demonstrating classification using photometric measurements alone.

A separate experiment excluding redshift could be performed to evaluate the models using only photometric and positional features.
