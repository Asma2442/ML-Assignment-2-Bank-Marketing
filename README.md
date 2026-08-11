
# Bank Marketing Classification

## 1. Problem Statement

The objective of this project is to build machine learning classification
models that predict whether a bank customer will subscribe to a term deposit
based on customer and marketing campaign information.

The project demonstrates an end-to-end machine learning workflow including:

- Dataset preparation
- Data preprocessing
- Training multiple classification models
- Model evaluation using multiple performance metrics
- Model comparison
- Saving trained models as pipelines
- Building an interactive Streamlit application
- Deploying the application for model demonstration


## 2. Dataset Description

The Bank Marketing dataset is obtained from the UCI Machine Learning
Repository.

The dataset contains information related to direct marketing campaigns
conducted by a Portuguese banking institution.

The classification target is:

- `y = yes` - customer subscribed to a term deposit
- `y = no` - customer did not subscribe to a term deposit

The dataset contains 45,211 instances and 16 input features.

### Input Features

The 16 input features used in this project are:

1. age
2. job
3. marital
4. education
5. default
6. balance
7. housing
8. loan
9. contact
10. day
11. month
12. duration
13. campaign
14. pdays
15. previous
16. poutcome

The target variable is:

- `y`

Dataset source:

https://archive.ics.uci.edu/dataset/222/bank%2B


## 3. Machine Learning Models

Five classification models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier
4. Gaussian Naive Bayes
5. Random Forest Classifier

The models were implemented using scikit-learn pipelines so that the
preprocessing steps are applied consistently during training and prediction.


## 4. Evaluation Metrics

Each model was evaluated using the following six metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)


## 5. Model Comparison

The following results were obtained on the test dataset.

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9012 | 0.9056 | 0.6445 | 0.3478 | 0.4518 | 0.4261 |
| Decision Tree | 0.8746 | 0.7015 | 0.4649 | 0.4754 | 0.4701 | 0.3990 |
| K-Nearest Neighbors | 0.8962 | 0.8277 | 0.5990 | 0.3403 | 0.4340 | 0.4001 |
| Gaussian Naive Bayes | 0.8548 | 0.8101 | 0.4059 | 0.5198 | 0.4559 | 0.3774 |
| Random Forest | 0.9045 | 0.9263 | 0.6506 | 0.3960 | 0.4924 | 0.4597 |


## 6. Observations on Model Performance

### Logistic Regression

Logistic Regression achieved an accuracy of 90.12% and an AUC of 0.9056.
It also achieved a relatively high precision of 0.6445.

The model provides a strong baseline and performs consistently across
the evaluation metrics.


### Decision Tree

The Decision Tree achieved an accuracy of 87.46% and an AUC of 0.7015.

Its recall of 0.4754 is higher than Logistic Regression, but its overall
performance is lower across several other metrics.


### K-Nearest Neighbors

K-Nearest Neighbors achieved an accuracy of 89.62% and an AUC of 0.8277.

Its performance is reasonable, but it does not outperform Logistic
Regression or Random Forest on the overall evaluation metrics.


### Gaussian Naive Bayes

Gaussian Naive Bayes achieved an accuracy of 85.48%.

However, it achieved the highest recall among the five models at 0.5198.
This means it identified a larger proportion of the positive class than
the other models.

Its precision and MCC were comparatively lower.


### Random Forest

Random Forest achieved the highest accuracy at 90.45%, the highest AUC
at 0.9263, the highest precision at 0.6506, the highest F1 Score at
0.4924, and the highest MCC at 0.4597.

Therefore, Random Forest provides the strongest overall performance on
this test dataset based on the evaluated metrics.

However, Gaussian Naive Bayes achieved higher recall than Random Forest.
Therefore, if identifying as many positive customers as possible is the
primary objective, recall should also be considered when selecting the
model.


## 7. Streamlit Application

An interactive Streamlit application was developed to demonstrate the
trained models.

The application provides:

- CSV test-data upload
- Model selection dropdown
- Predictions
- Model comparison table
- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC
- Confusion matrix
- Classification report

The application uses the saved trained model pipelines from the `models`
directory.


## 8. Project Structure

```text
Bank-Marketing-Classification/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
└── models/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── k-nearest_neighbors.pkl
    ├── gaussian_naive_bayes.pkl
    └── random_forest.pkl
